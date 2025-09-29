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
#[allow(unused_imports)]
use ort::Environment;
#[allow(unused_imports)]
use ndarray::Array4;

async fn stream_handler(_req: HttpRequest) -> impl Responder {
    let boundary = "boundarydonotcross";
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();

    HttpResponse::Ok()
        .append_header(("Content-Type", format!("multipart/x-mixed-replace; boundary={}", boundary)))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            loop {
                let mut frame = Mat::default();
                cam.read(&mut frame).unwrap();
                if frame.empty() {
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

// 骨格（手）推定のMJPEGストリーム。/stream はそのまま、こちらは別ルート。
async fn hand_stream_handler(_req: HttpRequest) -> impl Responder {
    let boundary = "boundarydonotcross";

    // カメラオープン
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();

    // モデル読み込み（簡便化のため各接続ごとに初期化）
    // 実運用では共有ステートに載せて再利用するのが望ましい
    let env = std::sync::Arc::new(Environment::builder().with_name("hand").build().unwrap());
    let _detector = env
        .new_session_builder()
        .unwrap()
        .with_model_from_file("/home/matsu/models/MediaPipeHandDetector.onnx")
        .unwrap();
    let _landmark = env
        .new_session_builder()
        .unwrap()
        .with_model_from_file("/home/matsu/models/hand_landmark_sparse_Nx3x224x224.onnx")
        .unwrap();

    HttpResponse::Ok()
        .append_header((
            "Content-Type",
            format!("multipart/x-mixed-replace; boundary={}", boundary),
        ))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            loop {
                let mut frame = Mat::default();
                cam.read(&mut frame).unwrap();
                if frame.empty() {
                    continue;
                }

                // 前処理テンソル（例: 224x224）
                let _input = preprocess(&frame, 224);

                // TODO: _input を detector に供給して手の検出ボックスを得る
                // TODO: 検出ボックスに基づき切り出し、landmark に供給
                // TODO: 出力21点を frame に描画（骨格ラインも）

                // ひとまずプレースホルダ表示（エンドポイントの有効性確認用）
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
                imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::<i32>::new()).unwrap();

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
        })
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
