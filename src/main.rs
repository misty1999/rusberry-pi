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

// モデル入力サイズ (MediaPipe Person DetectorはNHWC想定)
const DETECTOR_SIZE: i32 = 256;

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

// ========== 前処理 (NHWC) ==========
fn preprocess_nhwc(frame: &Mat, size: i32) -> Array4<f32> {
    let mut resized = Mat::default();
    imgproc::resize(&frame, &mut resized, Size::new(size, size), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0).unwrap();

    let data = rgb.data_bytes().unwrap();
    let mut input = Array4::<f32>::zeros((1, size as usize, size as usize, 3));
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

// Detector 出力 [1, N, 18] のフラット Vec<f32> から最大矩形を選ぶ
fn decode_best_detection(det: &[f32], cols: i32, rows: i32) -> Option<Rect> {
    const ELEM: usize = 18;
    if det.len() < ELEM { return None; }
    let count = det.len() / ELEM;

    let mut best_idx = None;
    let mut best_area = -1.0f32;

    for i in 0..count {
        let off = i * ELEM;
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
    Some(Rect::new(x, y, w.round() as i32, h.round() as i32))
}

// ========== 人物検出付き MJPEG ==========
async fn person_stream_handler(req: HttpRequest) -> Result<HttpResponse, actix_web::Error> {
    println!("[person] リクエスト受信: {}", req.path());
    let boundary = "boundarydonotcross";

    // カメラ
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .map_err(actix_web::error::ErrorInternalServerError)?;
    match cam.is_opened() {
        Ok(true) => println!("[person] カメラをオープンしました"),
        Ok(false) => println!("[person] カメラを開けませんでした"),
        Err(e) => println!("[person] カメラ状態取得エラー: {e}"),
    }

    // 人物検出モデル
    let mut detector: Session = SessionBuilder::new()
        .map_err(actix_web::error::ErrorInternalServerError)?
        .commit_from_file("/home/matsu/models/person_detector.onnx")
        .map_err(actix_web::error::ErrorInternalServerError)?;
    println!("[person] Detectorモデルをロードしました");
    println!("[person] 入力: {:?}, 出力: {:?}", detector.inputs, detector.outputs);

    Ok(HttpResponse::Ok()
        .append_header(("Content-Type", format!("multipart/x-mixed-replace; boundary={}", boundary)))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            loop {
                let mut frame = Mat::default();
                if cam.read(&mut frame).is_err() || frame.empty() {
                    continue;
                }

                // Detector 前処理 (NHWC) & 推論
                let det_in = preprocess_nhwc(&frame, DETECTOR_SIZE);
                let det_shape: Vec<usize> = det_in.shape().to_vec();
                let det_data: Vec<f32> = det_in.into_raw_vec();
                let det_val = match Value::from_array((det_shape, det_data)) {
                    Ok(v) => v, Err(_) => continue,
                };
                let det_outs = match detector.run([SessionInputValue::from(det_val)]) {
                    Ok(o) => o, Err(_) => continue,
                };

                if let Ok((shape, out_data)) = det_outs[0].try_extract_tensor::<f32>() {
                    println!("[person] 出力 shape = {:?}", shape);
                    let cols = frame.cols();
                    let rows = frame.rows();
                    if let Some(r) = decode_best_detection(&out_data, cols, rows) {
                        if let Some(r) = clamp_rect(r, cols, rows) {
                            let _ = imgproc::rectangle(
                                &mut frame,
                                r,
                                Scalar::new(0.0, 255.0, 0.0, 0.0),
                                2,
                                imgproc::LINE_8,
                                0,
                            );
                        }
                    }
                }

                // JPEG 送出
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
            .route("/person_stream", web::get().to(person_stream_handler))
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
