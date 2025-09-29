use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use opencv::{
    core::{self, Point, Scalar, Size, Rect},
    imgcodecs, imgproc, prelude::*, videoio,
};
use std::io::Write;
use std::thread;
use std::time::Duration;

// ort + ndarray は骨格推定エンドポイント用。
// まだ推論の実装を入れ切らないため、未使用警告を抑制します。
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;
use ort::session::SessionInputValue;
use ort::session::builder::SessionBuilder;

// モデル入力サイズ（Detectorは256、Landmarkは224想定）
const DETECTOR_SIZE: i32 = 256;
const LANDMARK_SIZE: i32 = 224;

async fn stream_handler(req: HttpRequest) -> impl Responder {
    println!("[stream] リクエスト受信: {}", req.path());
    let boundary = "boundarydonotcross";
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();
    match cam.is_opened() {
        Ok(true) => println!("[stream] カメラをオープンしました"),
        Ok(false) => println!("[stream] カメラを開けませんでした"),
        Err(e) => println!("[stream] カメラ状態取得エラー: {e}"),
    }

    HttpResponse::Ok()
        .append_header(("Content-Type", format!("multipart/x-mixed-replace; boundary={}", boundary)))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            let mut frame_count: u64 = 0;
            loop {
                let mut frame = Mat::default();
                if let Err(e) = cam.read(&mut frame) {
                    println!("[stream] フレーム読込エラー: {e}");
                    continue;
                }
                if frame.empty() {
                    println!("[stream] 空フレームをスキップ");
                    continue;
                }
                frame_count += 1;

                // === ここから輪郭検出処理 ===

                // グレースケール化
                let mut gray = Mat::default();
                imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();

                // 二値化
                let mut thresh = Mat::default();
                imgproc::threshold(&gray, &mut thresh, 100.0, 255.0, imgproc::THRESH_BINARY).unwrap();

                // 輪郭抽出（非推奨エイリアスをやめ、コア型を使用）
                let mut contours: core::Vector<core::Vector<Point>> = core::Vector::new();
                imgproc::find_contours(
                    &thresh,
                    &mut contours,
                    imgproc::RETR_EXTERNAL,
                    imgproc::CHAIN_APPROX_SIMPLE,
                    Point::new(0, 0),
                ).unwrap();
                println!("[stream] 輪郭数: {} (frame #{})", contours.len(), frame_count);

                // 輪郭描画
                imgproc::draw_contours(
                    &mut frame,
                    &contours,
                    -1, // 全部描画
                    Scalar::new(0.0, 255.0, 0.0, 0.0), // 緑色
                    2,
                    imgproc::LINE_8,
                    &core::no_array(),
                    i32::MAX,
                    Point::new(0, 0),
                ).unwrap();

                // === ここまで ===

                // JPEG エンコード（typesエイリアスではなくcore::Vectorを使用）
                let mut buf: core::Vector<u8> = core::Vector::new();
                if let Err(e) = imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::<i32>::new()) {
                    println!("[stream] JPEGエンコード失敗: {e}");
                    continue;
                }

                // multipart のフォーマットにして送信
                let mut data = Vec::new();
                write!(data, "--{}\r\n", boundary).unwrap();
                write!(data, "Content-Type: image/jpeg\r\n").unwrap();
                write!(data, "Content-Length: {}\r\n\r\n", buf.len()).unwrap();
                // OpenCVのVector<u8>は&[u8]ではないため、適切に変換して追記
                data.extend_from_slice(&buf.to_vec());
                data.extend_from_slice(b"\r\n");

                yield Ok::<_, actix_web::Error>(web::Bytes::from(data));
                thread::sleep(Duration::from_millis(100)); // 10fps
            }
        })
}

// 画像前処理: (1, 3, size, size) のRGB正規化テンソルに変換
#[allow(dead_code)]
fn preprocess(frame: &Mat, size: i32) -> Array4<f32> {
    let mut resized = Mat::default();
    imgproc::resize(
        &frame,
        &mut resized,
        Size::new(size, size),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .unwrap();

    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0).unwrap();

    let data = rgb.data_bytes().unwrap();
    let mut input = Array4::<f32>::zeros((1, 3, size as usize, size as usize));

    for y in 0..size {
        for x in 0..size {
            let base = ((y * size + x) * 3) as usize;
            input[[0, 0, y as usize, x as usize]] = data[base] as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = data[base + 1] as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = data[base + 2] as f32 / 255.0;
        }
    }
    input
}

// ROI切り出しの矩形を画像サイズに収める
fn clamp_rect(mut rect: Rect, cols: i32, rows: i32) -> Option<Rect> {
    if rect.x < 0 { rect.width += rect.x; rect.x = 0; }
    if rect.y < 0 { rect.height += rect.y; rect.y = 0; }
    if rect.x >= cols || rect.y >= rows { return None; }
    if rect.x + rect.width > cols { rect.width = cols - rect.x; }
    if rect.y + rect.height > rows { rect.height = rows - rect.y; }
    if rect.width <= 0 || rect.height <= 0 { return None; }
    Some(rect)
}

// BlazePalm系を想定: [cx, cy, w, h, kp0x, kp0y, ... kp6x, kp6y] の18要素、値は0..1正規化と仮定
fn decode_best_detection(det: &[f32], cols: i32, rows: i32) -> Option<(Rect, Vec<Point>)> {
    if det.len() % 18 != 0 { return None; }
    let mut best_idx = None;
    let mut best_area = -1.0f32;
    let count = det.len() / 18;
    for i in 0..count {
        let off = i * 18;
        let cx = det[off + 0].clamp(0.0, 1.0);
        let cy = det[off + 1].clamp(0.0, 1.0);
        let w = det[off + 2].abs();
        let h = det[off + 3].abs();
        let area = w * h;
        if area > best_area { best_area = area; best_idx = Some(i); }
    }
    let i = best_idx?;
    let off = i * 18;
    let cx = det[off + 0].clamp(0.0, 1.0) * cols as f32;
    let cy = det[off + 1].clamp(0.0, 1.0) * rows as f32;
    let w = (det[off + 2].abs() * cols as f32).max(1.0);
    let h = (det[off + 3].abs() * rows as f32).max(1.0);
    let x = (cx - 0.5 * w).round() as i32;
    let y = (cy - 0.5 * h).round() as i32;
    let rect = Rect::new(x, y, w.round() as i32, h.round() as i32);

    // 7キーポイントを画像座標に変換（0..1仮定）
    let mut kps = Vec::with_capacity(7);
    for k in 0..7 { // kp0..kp6
        let kx = det[off + 4 + k * 2].clamp(0.0, 1.0) * cols as f32;
        let ky = det[off + 4 + k * 2 + 1].clamp(0.0, 1.0) * rows as f32;
        kps.push(Point::new(kx.round() as i32, ky.round() as i32));
    }

    Some((rect, kps))
}

// 画像上に21点の手スケルトンを描画（MediaPipe標準の接続を簡略）
fn draw_hand_landmarks(frame: &mut Mat, pts: &[(i32, i32)]) {
    let color = Scalar::new(0.0, 200.0, 255.0, 0.0);
    for &(x, y) in pts {
        imgproc::circle(frame, Point::new(x, y), 3, color, -1, imgproc::LINE_8, 0).ok();
    }
    let edges: &[(usize, usize)] = &[
        // 手首→各指の起点
        (0, 1), (1, 2), (2, 3), (3, 4),      // 親指
        (0, 5), (5, 6), (6, 7), (7, 8),      // 人差し指
        (0, 9), (9,10), (10,11), (11,12),    // 中指
        (0,13), (13,14), (14,15), (15,16),   // 薬指
        (0,17), (17,18), (18,19), (19,20),   // 小指
    ];
    for &(a, b) in edges {
        if a < pts.len() && b < pts.len() {
            let pa = Point::new(pts[a].0, pts[a].1);
            let pb = Point::new(pts[b].0, pts[b].1);
            imgproc::line(frame, pa, pb, color, 2, imgproc::LINE_8, 0).ok();
        }
    }
}

async fn hand_stream_handler(req: HttpRequest) -> Result<HttpResponse, actix_web::Error> {
    println!("[hand] リクエスト受信: {}", req.path());
    let boundary = "boundarydonotcross";

    // カメラオープン
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .map_err(actix_web::error::ErrorInternalServerError)?;
    match cam.is_opened() {
        Ok(true) => println!("[hand] カメラをオープンしました"),
        Ok(false) => println!("[hand] カメラを開けませんでした"),
        Err(e) => println!("[hand] カメラ状態取得エラー: {e}"),
    }

    // --- モデルは最初にロードして再利用する ---
    let mut detector: Session = SessionBuilder::new()
        .map_err(actix_web::error::ErrorInternalServerError)?
        .commit_from_file("/home/matsu/models/MediaPipeHandDetector.onnx")
        .map_err(actix_web::error::ErrorInternalServerError)?;
    println!("[hand] Detectorモデルをロードしました");

    let landmark: Session = SessionBuilder::new()
        .map_err(actix_web::error::ErrorInternalServerError)?
        .commit_from_file("/home/matsu/models/hand_landmark_sparse_Nx3x224x224.onnx")
        .map_err(actix_web::error::ErrorInternalServerError)?;
    println!("[hand] Landmarkモデルをロードしました");

    Ok(HttpResponse::Ok()
        .append_header((
            "Content-Type",
            format!("multipart/x-mixed-replace; boundary={}", boundary),
        ))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            let mut frame_count: u64 = 0;
            loop {
                let mut frame = Mat::default();
                if let Err(e) = cam.read(&mut frame) {
                    println!("[hand] フレーム読込エラー: {e}");
                    yield Err(actix_web::error::ErrorInternalServerError(e));
                    continue;
                }
                if frame.empty() {
                    println!("[hand] 空フレームをスキップ");
                    continue;
                }
                frame_count += 1;

                // --- 入力テンソル作成（Detectorは256x256）---
                let input: Array4<f32> = preprocess(&frame, DETECTOR_SIZE);
                let shape: Vec<usize> = input.shape().to_vec();
                let data: Vec<f32> = input.into_raw_vec();
                let input_value = match Value::from_array((shape, data)) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("[hand] 入力テンソル作成エラー: {e}");
                        yield Err(actix_web::error::ErrorInternalServerError(e));
                        continue;
                    }
                };

                // --- 推論実行 ---
                let outputs = match detector.run([SessionInputValue::from(input_value)]) {
                    Ok(o) => o,
                    Err(e) => {
                        println!("[hand] 推論エラー: {e}");
                        yield Err(actix_web::error::ErrorInternalServerError(e));
                        continue;
                    }
                };

                // --- 手検出出力のデコード（仮定ベース）---
                if let Ok((out_shape, out_data)) = outputs[0].try_extract_tensor::<f32>() {
                    println!("[hand] 出力shape: {:?}", out_shape);
                    let cols = frame.cols();
                    let rows = frame.rows();
                    if let Some((rect, kps)) = decode_best_detection(&out_data, cols, rows) {
                        if let Some(r) = clamp_rect(rect, cols, rows) {
                            // ROI抽出
                            let roi = Mat::roi(&frame, r).unwrap_or_else(|_| frame.clone());

                            // Landmark前処理（224x224）
                            let l_input: Array4<f32> = preprocess(&roi, LANDMARK_SIZE);
                            let l_shape: Vec<usize> = l_input.shape().to_vec();
                            let l_data: Vec<f32> = l_input.into_raw_vec();
                            match Value::from_array((l_shape, l_data)) {
                                Ok(l_val) => {
                                    match landmark.run([SessionInputValue::from(l_val)]) {
                                        Ok(l_outs) => {
                                            if let Ok((_ls, ldata)) = l_outs[0].try_extract_tensor::<f32>() {
                                                // 21*3 = 63 を想定
                                                let n = ldata.len().min(63);
                                                if n >= 63 {
                                                    let mut pts: Vec<(i32,i32)> = Vec::with_capacity(21);
                                                    for j in 0..21 {
                                                        let x = ldata[j*3 + 0].clamp(0.0, 1.0) * r.width as f32 + r.x as f32;
                                                        let y = ldata[j*3 + 1].clamp(0.0, 1.0) * r.height as f32 + r.y as f32;
                                                        pts.push((x.round() as i32, y.round() as i32));
                                                    }
                                                    // スケルトン描画
                                                    draw_hand_landmarks(&mut frame, &pts);
                                                } else {
                                                    println!("[hand] Landmark出力が短い: {}", ldata.len());
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            println!("[hand] Landmark推論エラー: {e}");
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("[hand] Landmark入力生成エラー: {e}");
                                }
                            }

                            // 検出矩形とキーポイントも描画
                            imgproc::rectangle(&mut frame, r, Scalar::new(255.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0).ok();
                            for p in kps { imgproc::circle(&mut frame, p, 2, Scalar::new(0.0, 255.0, 255.0, 0.0), -1, imgproc::LINE_8, 0).ok(); }
                        }
                    }
                }

                // TODO: 検出結果を使ってROIを切り出し、landmarkモデルに入力
                // TODO: 出力21点をframeに描画

                // プレースホルダ表示
                imgproc::put_text(
                    &mut frame,
                    "Hand Pose",
                    Point::new(20, 40),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    Scalar::new(0.0, 255.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    false,
                ).ok();

                // JPEG エンコード
                let mut buf: core::Vector<u8> = core::Vector::new();
                if let Err(e) = imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::<i32>::new()) {
                    println!("[hand] JPEGエンコード失敗: {e}");
                    yield Err(actix_web::error::ErrorInternalServerError(e));
                    continue;
                }

                // multipart 形式で送信
                let mut data = Vec::new();
                write!(data, "--{}\r\n", boundary).unwrap();
                write!(data, "Content-Type: image/jpeg\r\n").unwrap();
                write!(data, "Content-Length: {}\r\n\r\n", buf.len()).unwrap();
                data.extend_from_slice(&buf.to_vec());
                data.extend_from_slice(b"\r\n");

                yield Ok::<_, actix_web::Error>(web::Bytes::from(data));
                thread::sleep(Duration::from_millis(100)); // 10fps
            }
        }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/stream", web::get().to(stream_handler))
            .route("/hand_stream", web::get().to(hand_stream_handler))
    })
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}
