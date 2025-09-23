use opencv::{
    core,
    highgui,
    imgproc,
    prelude::*,
    videoio,
    types,
};

fn main() -> opencv::Result<()> {
    // カメラを開く (0番 = /dev/video0)
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("カメラが開けませんでした");
    }

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        // グレースケールに変換
        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // 二値化
        let mut thresh = Mat::default();
        imgproc::threshold(&gray, &mut thresh, 100.0, 255.0, imgproc::THRESH_BINARY)?;

        // 輪郭検出
        let mut contours = types::VectorOfVectorOfPoint::new();
        imgproc::find_contours(
            &thresh,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            core::Point::new(0, 0),
        )?;

        // 輪郭を元の画像に描画
        imgproc::draw_contours(
            &mut frame,
            &contours,
            -1, // -1 = 全部描画
            core::Scalar::new(0.0, 255.0, 0.0, 0.0), // 緑色
            2,
            imgproc::LINE_8,
            &core::no_array(),
            i32::MAX,
            core::Point::new(0, 0),
        )?;

        // 映像を表示
        highgui::imshow("Hand Contours", &frame)?;

        // 10ms待ってキー押下チェック、ESCなら終了
        if highgui::wait_key(10)? == 27 {
            break;
        }
    }

    Ok(())
}
