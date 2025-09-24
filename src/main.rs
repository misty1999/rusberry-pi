use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use opencv::{
    core::{self, Point, Scalar},
    imgcodecs, imgproc, prelude::*, videoio,
};
use std::io::Write;
use std::thread;
use std::time::Duration;

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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/stream", web::get().to(stream_handler)))
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}
