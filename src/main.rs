use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use opencv::{
    core::{self, Point, Scalar, Size, Rect},
    imgcodecs, imgproc, prelude::*, videoio,
};
use std::io::Write;
use std::thread;
use std::time::Duration;

use ndarray::Array4;
use candle_core::{Device, Tensor, DType};
use candle_onnx::Model;

// ---- Model settings ----
const DETECTOR_SIZE: i32 = 224;       // [1,3,224,224]
const CONF_THRESH: f32 = 0.40;        // 表示しきい値
const MAX_DRAW: usize = 5;            // 最大検出数

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

                // 簡易輪郭描画
                let mut gray = Mat::default();
                imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0).ok();
                let mut thresh = Mat::default();
                imgproc::threshold(&gray, &mut thresh, 100.0, 255.0, imgproc::THRESH_BINARY).ok();
                let mut contours: core::Vector<core::Vector<Point>> = core::Vector::new();
                imgproc::find_contours(&thresh, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0)).ok();
                imgproc::draw_contours(&mut frame, &contours, -1, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, &core::no_array(), i32::MAX, Point::new(0, 0)).ok();

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
        })
}

// ========== 前処理 (NCHW) ==========
fn preprocess_nchw(frame: &Mat, size: i32) -> Array4<f32> {
    let mut resized = Mat::default();
    imgproc::resize(&frame, &mut resized, Size::new(size, size), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

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

// [boxes(=B*D), scores(=B)] -> (Rect, conf) のリスト
fn decode_persons_with_scores(
    boxes: &[f32],
    scores: &[f32],
    boxes_dim: usize,
    cols: i32,
    rows: i32,
    conf_thresh: f32,
) -> Vec<(Rect, f32)> {
    let anchors_boxes = boxes.len() / boxes_dim;
    let anchors_scores = scores.len();
    let n = anchors_boxes.min(anchors_scores);
    let mut out: Vec<(Rect, f32)> = Vec::new();

    for i in 0..n {
        let conf = scores[i];
        if conf < conf_thresh { continue; }

        let off = i * boxes_dim;
        let cx = boxes[off + 0] * cols as f32;
        let cy = boxes[off + 1] * rows as f32;
        let w  = boxes[off + 2] * cols as f32;
        let h  = boxes[off + 3] * rows as f32;

        let mut rect = Rect::new(
            (cx - 0.5 * w).round() as i32,
            (cy - 0.5 * h).round() as i32,
            w.max(1.0).round() as i32,
            h.max(1.0).round() as i32,
        );
        if let Some(r) = clamp_rect(rect, cols, rows) {
            rect = r;
        }
        out.push((rect, conf));
    }

    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if out.len() > MAX_DRAW {
        out.truncate(MAX_DRAW);
    }
    out
}

// ========== 人物検出付き MJPEG ==========
async fn person_stream_handler(req: HttpRequest) -> Result<HttpResponse, actix_web::Error> {
    println!("[person] リクエスト受信: {}", req.path());
    let boundary = "boundarydonotcross";

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .map_err(actix_web::error::ErrorInternalServerError)?;
    match cam.is_opened() {
        Ok(true) => println!("[person] カメラをオープンしました"),
        Ok(false) => println!("[person] カメラを開けませんでした"),
        Err(e) => println!("[person] カメラ状態取得エラー: {e}"),
    }

    // Candle モデル読み込み
    let device = Device::Cpu;
    let model = Model::load("/home/matsu/models/person_detector.onnx", &device)
        .map_err(actix_web::error::ErrorInternalServerError)?;
    println!("[person] Detectorモデルをロードしました");

    Ok(HttpResponse::Ok()
        .append_header(("Content-Type", format!("multipart/x-mixed-replace; boundary={}", boundary)))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            loop {
                let mut frame = Mat::default();
                if cam.read(&mut frame).is_err() || frame.empty() {
                    continue;
                }

                // 前処理
                let det_in = preprocess_nchw(&frame, DETECTOR_SIZE);
                let det_shape: Vec<usize> = det_in.shape().to_vec();
                let det_data: Vec<f32> = det_in.into_raw_vec();

                // Tensor 化
                let input = match Tensor::from_vec(
                    det_data,
                    (det_shape[0], det_shape[1], det_shape[2], det_shape[3]),
                    &device
                ) {
                    Ok(t) => t,
                    Err(e) => { println!("[person] 入力Tensor作成エラー: {e}"); continue; }
                };

                // 推論
                let outputs = match model.run(vec![input]) {
                    Ok(o) => o,
                    Err(e) => { println!("[person] 推論エラー: {e}"); continue; }
                };

                let mut boxes_data: Option<(Vec<f32>, usize)> = None;
                let mut scores_data: Option<Vec<f32>> = None;

                for (idx, out) in outputs.iter().enumerate() {
                    let shape = out.dims();
                    if shape.len() == 3 && shape[2] == 12 {
                        if let Ok(data) = out.to_vec1::<f32>() {
                            println!("[person] out[{idx}] -> boxes {:?}, len={}", shape, data.len());
                            boxes_data = Some((data, 12));
                        }
                    } else if shape.len() == 3 && shape[2] == 1 {
                        if let Ok(data) = out.to_vec1::<f32>() {
                            println!("[person] out[{idx}] -> scores {:?}", shape);
                            scores_data = Some(data);
                        }
                    } else {
                        println!("[person] out[{idx}] 未対応 shape = {:?}", shape);
                    }
                }

                if let (Some((boxes, dim)), Some(scores)) = (boxes_data, scores_data) {
                    let cols = frame.cols();
                    let rows = frame.rows();
                    let rects = decode_persons_with_scores(&boxes, &scores, dim, cols, rows, CONF_THRESH);

                    for (r, conf) in rects {
                        let _ = imgproc::rectangle(&mut frame, r, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0);
                        let label = format!("{:.2}", conf);
                        let _ = imgproc::put_text(
                            &mut frame,
                            &label,
                            Point::new(r.x.max(0), (r.y - 6).max(12)),
                            imgproc::FONT_HERSHEY_SIMPLEX,
                            0.6,
                            Scalar::new(0.0, 255.0, 0.0, 0.0),
                            2,
                            imgproc::LINE_8,
                            false,
                        );
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
                thread::sleep(Duration::from_millis(100));
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
