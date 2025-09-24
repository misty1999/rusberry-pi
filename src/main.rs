use actix_web::{middleware, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use opencv::{
    core::{self, Point, Scalar},
    imgcodecs, imgproc, prelude::*, videoio,
};
use std::io::Write;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use env_logger::Env;
use log::{debug, error, info, warn};

async fn stream_handler(req: HttpRequest) -> impl Responder {
    let boundary = "boundarydonotcross";
    info!("/stream 要求: peer={:?}", req.peer_addr());

    debug!("カメラを初期化します (index=0, backend=ANY)");
    let mut cam = match videoio::VideoCapture::new(0, videoio::CAP_ANY) {
        Ok(c) => c,
        Err(e) => {
            error!("カメラ初期化に失敗: {}", e);
            return HttpResponse::InternalServerError().body("failed to open camera");
        }
    };

    match cam.is_opened() {
        Ok(true) => info!("カメラ接続に成功"),
        Ok(false) => {
            error!("カメラが開けませんでした");
            return HttpResponse::InternalServerError().body("camera not opened");
        }
        Err(e) => {
            error!("カメラ状態取得に失敗: {}", e);
            return HttpResponse::InternalServerError().body("camera status error");
        }
    }

    HttpResponse::Ok()
        .append_header(("Content-Type", format!("multipart/x-mixed-replace; boundary={}", boundary)))
        .streaming::<_, actix_web::Error>(async_stream::stream! {
            loop {
                let loop_start = Instant::now();
                let mut frame = Mat::default();
                if let Err(e) = cam.read(&mut frame) {
                    error!("フレーム読込に失敗: {}", e);
                    break;
                }
                if frame.empty() {
                    debug!("空フレームを受信。スキップします");
                    continue;
                }

                // === ここから輪郭検出処理 ===

                // グレースケール化
                let mut gray = Mat::default();
                if let Err(e) = imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0) {
                    warn!("cvt_color 失敗: {}", e);
                    continue;
                }

                // 二値化
                let mut thresh = Mat::default();
                if let Err(e) = imgproc::threshold(&gray, &mut thresh, 100.0, 255.0, imgproc::THRESH_BINARY) {
                    warn!("threshold 失敗: {}", e);
                    continue;
                }

                // 輪郭抽出（非推奨エイリアスをやめ、コア型を使用）
                let mut contours: core::Vector<core::Vector<Point>> = core::Vector::new();
                if let Err(e) = imgproc::find_contours(
                    &thresh,
                    &mut contours,
                    imgproc::RETR_EXTERNAL,
                    imgproc::CHAIN_APPROX_SIMPLE,
                    Point::new(0, 0),
                ) {
                    warn!("find_contours 失敗: {}", e);
                    continue;
                }

                debug!("検出された輪郭数: {}", contours.len());

                // 輪郭描画
                if let Err(e) = imgproc::draw_contours(
                    &mut frame,
                    &contours,
                    -1, // 全部描画
                    Scalar::new(0.0, 255.0, 0.0, 0.0), // 緑色
                    2,
                    imgproc::LINE_8,
                    &core::no_array(),
                    i32::MAX,
                    Point::new(0, 0),
                ) {
                    warn!("draw_contours 失敗: {}", e);
                    continue;
                }

                // === ここまで ===

                // JPEG エンコード（typesエイリアスではなくcore::Vectorを使用）
                let mut buf: core::Vector<u8> = core::Vector::new();
                if let Err(e) = imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::<i32>::new()) {
                    warn!("JPEG エンコード失敗: {}", e);
                    continue;
                }
                debug!("JPEG サイズ: {} bytes", buf.len());

                // multipart のフォーマットにして送信
                let mut data = Vec::new();
                write!(data, "--{}\r\n", boundary).unwrap();
                write!(data, "Content-Type: image/jpeg\r\n").unwrap();
                write!(data, "Content-Length: {}\r\n\r\n", buf.len()).unwrap();
                // OpenCVのVector<u8>は&[u8]ではないため、適切に変換して追記
                data.extend_from_slice(&buf.to_vec());
                data.extend_from_slice(b"\r\n");

                yield Ok::<_, actix_web::Error>(web::Bytes::from(data));
                let elapsed = loop_start.elapsed();
                debug!("1 ループ処理時間: {:?}", elapsed);
                thread::sleep(Duration::from_millis(100)); // 10fps 相当のペース
            }
        })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // ロガー初期化（RUST_LOG 未設定なら info デフォルト）
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    info!("サーバー起動: http://0.0.0.0:8080/stream");

    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default())
            .route("/stream", web::get().to(stream_handler))
    })
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}
