use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use opencv::{
    core::{self, Point, Scalar, Size},
    imgcodecs, imgproc, prelude::*, videoio,
};
use std::io::Write;
use tokio::time::{sleep, Duration};

// ort + ndarray は骨格推定エンドポイント用。
// まだ推論の実装を入れ切らないため、未使用警告を抑制します。
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;
use ort::session::SessionInputValue;
use ort::session::builder::SessionBuilder;

async fn stream_handler(_req: HttpRequest) -> Result<HttpResponse, actix_web::Error> {
    let boundary = "boundarydonotcross";
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_V4L2)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok()
        .append_header(("Content-Type", format!("multipart/x-mixed-replace; boundary={}", boundary)))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            loop {
                let mut frame = Mat::default();
                if !cam.read(&mut frame).unwrap_or(false) || frame.empty() {
                    eprintln!("Frame capture failed");
                    continue;
                }

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
                imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::<i32>::new()).unwrap();

                // multipart のフォーマットにして送信
                let mut data = Vec::new();
                write!(data, "--{}\r\n", boundary).unwrap();
                write!(data, "Content-Type: image/jpeg\r\n").unwrap();
                write!(data, "Content-Length: {}\r\n\r\n", buf.len()).unwrap();
                // OpenCVのVector<u8>は&[u8]ではないため、適切に変換して追記
                data.extend_from_slice(&buf.to_vec());
                data.extend_from_slice(b"\r\n");

                yield Ok::<_, actix_web::Error>(web::Bytes::from(data));
                sleep(Duration::from_millis(100)).await; // 10fps
            }
        }))
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

async fn hand_stream_handler(_req: HttpRequest) -> Result<HttpResponse, actix_web::Error> {
    let boundary = "boundarydonotcross";

    // カメラオープン
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_V4L2)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    // --- モデルは最初にロードして再利用する ---
    let mut detector: Session = SessionBuilder::new()
        .map_err(actix_web::error::ErrorInternalServerError)?
        .commit_from_file("/home/pi/models/MediaPipeHandDetector.onnx")
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let landmark: Session = SessionBuilder::new()
        .map_err(actix_web::error::ErrorInternalServerError)?
        .commit_from_file("/home/pi/models/hand_landmark_sparse_Nx3x224x224.onnx")
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok()
        .append_header((
            "Content-Type",
            format!("multipart/x-mixed-replace; boundary={}", boundary),
        ))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            loop {
                let mut frame = Mat::default();
                if !cam.read(&mut frame).unwrap_or(false) || frame.empty() {
                    eprintln!("Frame capture failed");
                    continue;
                }

                // --- 入力テンソル作成 ---
                let input: Array4<f32> = preprocess(&frame, 224);
                let shape: Vec<usize> = input.shape().to_vec();
                let data: Vec<f32> = input.into_raw_vec();
                let input_value = match Value::from_array((shape, data)) {
                    Ok(v) => v,
                    Err(e) => {
                        yield Err(actix_web::error::ErrorInternalServerError(e));
                        continue;
                    }
                };

                // --- 推論実行 ---
                let outputs = match detector.run([SessionInputValue::from(input_value)]) {
                    Ok(o) => o,
                    Err(e) => {
                        yield Err(actix_web::error::ErrorInternalServerError(e));
                        continue;
                    }
                };

                // --- 出力確認 ---
                if let Ok((out_shape, out_data)) = outputs[0].try_extract_tensor::<f32>() {
                    dbg!(out_shape);
                    dbg!(out_data.get(0));
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
                sleep(Duration::from_millis(100)).await; // 10fps
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
