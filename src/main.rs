use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use opencv::{
    core::{self, Point, Scalar, Size, Rect},
    imgcodecs, imgproc, prelude::*, videoio,
};
use std::io::Write;
use std::thread;
use std::time::Duration;

use ndarray::Array4;
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::session::builder::SessionBuilder;
use ort::value::Value;

// モデル入力サイズ（Detectorは256 NHWC、Landmarkは224 NCHW 想定）
const DETECTOR_SIZE: i32 = 256;
const LANDMARK_SIZE: i32 = 224;

// ========== 汎用 MJPEG カメラ配信（デバッグ用） ==========
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

                // 輪郭可視化（簡易）
                let mut gray = Mat::default();
                imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0).ok();
                let mut thresh = Mat::default();
                imgproc::threshold(&gray, &mut thresh, 100.0, 255.0, imgproc::THRESH_BINARY).ok();
                let mut contours: core::Vector<core::Vector<Point>> = core::Vector::new();
                imgproc::find_contours(&thresh, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0)).ok();
                imgproc::draw_contours(&mut frame, &contours, -1, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, &core::no_array(), i32::MAX, Point::new(0, 0)).ok();

                // JPEG で 1 フレーム送出
                let mut buf: core::Vector<u8> = core::Vector::new();
                if imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::<i32>::new()).is_err() {
                    continue;
                }

                let mut data = Vec::new();
                let _ = write!(data, "--{}\r\n", boundary);
                let _ = write!(data, "Content-Type: image/jpeg\r\n");
                let _ = write!(data, "Content-Length: {}\r\n\r\n", buf.len());
                data.extend_from_slice(&buf.to_vec());
                data.extend_from_slice(b"\r\n");

                yield Ok::<_, actix_web::Error>(web::Bytes::from(data));
                thread::sleep(Duration::from_millis(100)); // 10fps
            }
        })
}

// ========== 前処理 ==========
/* Detector 用: NHWC (1,H,W,3) RGB [0..1] */
fn preprocess_nhwc(frame: &Mat, size: i32) -> Array4<f32> {
    let mut resized = Mat::default();
    imgproc::resize(&frame, &mut resized, Size::new(size, size), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0).unwrap();

    let data = rgb.data_bytes().unwrap();
    let mut input = Array4::<f32>::zeros((1, size as usize, size as usize, 3));
    // NHWC: (1, y, x, c)
    for y in 0..size {
        for x in 0..size {
            let base = ((y * size + x) * 3) as usize;
            input[[0, y as usize, x as usize, 0]] = data[base] as f32 / 255.0;
            input[[0, y as usize, x as usize, 1]] = data[base + 1] as f32 / 255.0;
            input[[0, y as usize, x as usize, 2]] = data[base + 2] as f32 / 255.0;
        }
    }
    input
}

/* Landmark 用: NCHW (1,3,H,W) RGB [0..1] */
fn preprocess_nchw(frame: &Mat, size: i32) -> Array4<f32> {
    let mut resized = Mat::default();
    imgproc::resize(&frame, &mut resized, Size::new(size, size), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0).unwrap();

    let data = rgb.data_bytes().unwrap();
    let mut input = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    // NCHW: (1, c, y, x)
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

// 画像サイズに収める
fn clamp_rect(mut rect: Rect, cols: i32, rows: i32) -> Option<Rect> {
    if rect.x < 0 { rect.width += rect.x; rect.x = 0; }
    if rect.y < 0 { rect.height += rect.y; rect.y = 0; }
    if rect.x >= cols || rect.y >= rows { return None; }
    if rect.x + rect.width > cols { rect.width = cols - rect.x; }
    if rect.y + rect.height > rows { rect.height = rows - rect.y; }
    if rect.width <= 0 || rect.height <= 0 { return None; }
    Some(rect)
}

// Detector 出力 [1, 2944, 18] のフラット Vec<f32> から最大矩形を選ぶ
fn decode_best_detection(det: &[f32], cols: i32, rows: i32) -> Option<(Rect, Vec<Point>)> {
    const ELEM: usize = 18;
    if det.len() < ELEM { return None; }
    let count = det.len() / ELEM;

    let mut best_idx = None;
    let mut best_area = -1.0f32;

    for i in 0..count {
        let off = i * ELEM;
        let cx = det[off + 0].clamp(0.0, 1.0);
        let cy = det[off + 1].clamp(0.0, 1.0);
        let w  = det[off + 2].abs();
        let h  = det[off + 3].abs();
        let area = w * h;
        if area > best_area {
            best_area = area;
            best_idx = Some(i);
        }
    }

    let i = best_idx?;
    let off = i * ELEM;

    let cx = det[off + 0].clamp(0.0, 1.0) * cols as f32;
    let cy = det[off + 1].clamp(0.0, 1.0) * rows as f32;
    let w  = (det[off + 2].abs() * cols as f32).max(1.0);
    let h  = (det[off + 3].abs() * rows as f32).max(1.0);
    let x  = (cx - 0.5 * w).round() as i32;
    let y  = (cy - 0.5 * h).round() as i32;
    let rect = Rect::new(x, y, w.round() as i32, h.round() as i32);

    // 7 keypoints
    let mut kps = Vec::with_capacity(7);
    for k in 0..7 {
        let kx = det[off + 4 + k*2].clamp(0.0, 1.0) * cols as f32;
        let ky = det[off + 4 + k*2 + 1].clamp(0.0, 1.0) * rows as f32;
        kps.push(Point::new(kx.round() as i32, ky.round() as i32));
    }
    Some((rect, kps))
}

// 21点骨格の簡易描画
fn draw_hand_landmarks(frame: &mut Mat, pts: &[(i32, i32)]) {
    let color = Scalar::new(0.0, 200.0, 255.0, 0.0);
    for &(x, y) in pts {
        let _ = imgproc::circle(frame, Point::new(x, y), 3, color, -1, imgproc::LINE_8, 0);
    }
    let edges: &[(usize, usize)] = &[
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
            let _ = imgproc::line(frame, pa, pb, color, 2, imgproc::LINE_8, 0);
        }
    }
}

// ========== 推論付き MJPEG ==========
async fn hand_stream_handler(req: HttpRequest) -> Result<HttpResponse, actix_web::Error> {
    println!("[hand] リクエスト受信: {}", req.path());
    let boundary = "boundarydonotcross";

    // カメラ
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .map_err(actix_web::error::ErrorInternalServerError)?;
    match cam.is_opened() {
        Ok(true) => println!("[hand] カメラをオープンしました"),
        Ok(false) => println!("[hand] カメラを開けませんでした"),
        Err(e) => println!("[hand] カメラ状態取得エラー: {e}"),
    }

    // モデル
    let mut detector: Session = SessionBuilder::new()
        .map_err(actix_web::error::ErrorInternalServerError)?
        .commit_from_file("/home/matsu/models/MediaPipeHandDetector.onnx")
        .map_err(actix_web::error::ErrorInternalServerError)?;
    println!("[hand] Detectorモデルをロードしました");

    let mut landmark: Session = SessionBuilder::new()
        .map_err(actix_web::error::ErrorInternalServerError)?
        .commit_from_file("/home/matsu/models/hand_landmark_sparse_Nx3x224x224.onnx")
        .map_err(actix_web::error::ErrorInternalServerError)?;
    println!("[hand] Landmarkモデルをロードしました");

    Ok(HttpResponse::Ok()
        .append_header(("Content-Type", format!("multipart/x-mixed-replace; boundary={}", boundary)))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            loop {
                let mut frame = Mat::default();
                if let Err(e) = cam.read(&mut frame) {
                    println!("[hand] フレーム読込エラー: {e}");
                    continue; // ストリームは維持
                }
                if frame.empty() { continue; }

                // ===== Detector 前処理 (NHWC) & 推論 =====
                let det_in = preprocess_nhwc(&frame, DETECTOR_SIZE);
                let det_shape: Vec<usize> = det_in.shape().to_vec();
                let det_data: Vec<f32> = det_in.into_raw_vec();

                let det_val = match Value::from_array((det_shape, det_data)) {
                    Ok(v) => v, Err(e) => { println!("[hand] Detector入力作成エラー: {e}"); continue; }
                };
                let det_outs = match detector.run([SessionInputValue::from(det_val)]) {
                    Ok(o) => o, Err(e) => { println!("[hand] Detector推論エラー: {e}"); continue; }
                };

                // 出力 [1,2944,18] を期待
                let mut rect_drawn = false;
                if let Ok((out_shape, out_data)) = det_outs[0].try_extract_tensor::<f32>() {
                    // println!("[hand] Detector 出力 shape = {:?}", out_shape); // 必要ならデバッグ表示
                    let cols = frame.cols();
                    let rows = frame.rows();
                    if let Some((rect, kps)) = decode_best_detection(&out_data, cols, rows) {
                        if let Some(r) = clamp_rect(rect, cols, rows) {
                            // ROI 抽出
                            let roi = match Mat::roi(&frame, r) {
                                Ok(sub) => match sub.try_clone() {
                                    Ok(m) => m,
                                    Err(_) => frame.clone(),
                                },
                                Err(_) => frame.clone(),
                            };

                            // ===== Landmark 前処理 (NCHW) & 推論 =====
                            let lm_in = preprocess_nhwc(&roi, LANDMARK_SIZE);
                            let lm_shape: Vec<usize> = lm_in.shape().to_vec();
                            let lm_data: Vec<f32> = lm_in.into_raw_vec();

                            if let Ok(lm_val) = Value::from_array((lm_shape, lm_data)) {
                                if let Ok(lm_outs) = landmark.run([SessionInputValue::from(lm_val)]) {
                                    if let Ok((_ls, ldata)) = lm_outs[0].try_extract_tensor::<f32>() {
                                        // 21*3 = 63 を想定 (x,y,z) だがここでは x,y のみ使用
                                        if ldata.len() >= 63 {
                                            let mut pts: Vec<(i32, i32)> = Vec::with_capacity(21);
                                            for j in 0..21 {
                                                let x = ldata[j*3 + 0].clamp(0.0, 1.0) * r.width as f32 + r.x as f32;
                                                let y = ldata[j*3 + 1].clamp(0.0, 1.0) * r.height as f32 + r.y as f32;
                                                pts.push((x.round() as i32, y.round() as i32));
                                            }
                                            draw_hand_landmarks(&mut frame, &pts);
                                        } else {
                                            println!("[hand] Landmark 出力が短い: {}", ldata.len());
                                        }
                                    }
                                }
                            }

                            // 検出の可視化
                            let _ = imgproc::rectangle(&mut frame, r, Scalar::new(255.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0);
                            for p in kps { let _ = imgproc::circle(&mut frame, p, 2, Scalar::new(0.0, 255.0, 255.0, 0.0), -1, imgproc::LINE_8, 0); }
                            rect_drawn = true;
                        }
                    }
                }

                let _ = imgproc::put_text(
                    &mut frame,
                    if rect_drawn { "Hand Pose (detected)" } else { "Hand Pose" },
                    Point::new(20, 40),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    Scalar::new(0.0, 255.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    false,
                );

                // JPEG 1フレーム送出
                let mut buf: core::Vector<u8> = core::Vector::new();
                if imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::<i32>::new()).is_err() {
                    continue;
                }
                let mut data = Vec::new();
                let _ = write!(data, "--{}\r\n", boundary);
                let _ = write!(data, "Content-Type: image/jpeg\r\n");
                let _ = write!(data, "Content-Length: {}\r\n\r\n", buf.len());
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
