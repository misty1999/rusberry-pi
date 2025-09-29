use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use opencv::{
    core::{self, Point, Scalar, Size},
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

                // --- 入力テンソル作成 ---
                let input: Array4<f32> = preprocess(&frame, 224);
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

                // --- 出力確認 ---
                if let Ok((out_shape, out_data)) = outputs[0].try_extract_tensor::<f32>() {
                    println!("[hand] 出力shape: {:?}", out_shape);
                    println!("[hand] 出力先頭要素: {:?}", out_data.get(0));
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
