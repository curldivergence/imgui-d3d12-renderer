use imgui::{im_str, FontConfig, FontSource};
use imgui_winit_support::{HiDpiMode, WinitPlatform};

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use rusty_d3d12::*;

use std::time::Instant;

const WINDOW_WIDTH: f64 = 760.0;
const WINDOW_HEIGHT: f64 = 760.0;


fn main() {
    let command_args = clap::App::new("ImGUI D3D12 Example")
    .arg(
        clap::Arg::with_name("breakonerr")
            .short("b")
            .takes_value(false)
            .help("Break on validation errors"),
    )
    .arg(
        clap::Arg::with_name("v")
            .short("v")
            .multiple(true)
            .help("Verbosity level"),
    )
    .arg(
        clap::Arg::with_name("debug")
            .short("d")
            .takes_value(false)
            .help("Use D3D debug layers"),
    )
    .get_matches();

    match command_args.occurrences_of("v") {
        0 => log_level = log::Level::Debug,
        1 | _ => log_level = log::Level::Trace,
    };


    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("imgui_dx11_renderer winit example")
        .with_inner_size(LogicalSize { width: WINDOW_WIDTH, height: WINDOW_HEIGHT })
        .build(&event_loop)
        .unwrap();
    let hwnd = if let RawWindowHandle::Windows(handle) = window.raw_window_handle() {
        handle.hwnd
    } else {
        unreachable!()
    };
    let (swapchain, device, context) = unsafe { create_device(hwnd.cast()) }.unwrap();
    let mut main_rtv = unsafe { create_render_target(&swapchain, &device) };
    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);

    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Rounded);

    let hidpi_factor = platform.hidpi_factor();
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(FontConfig { size_pixels: font_size, ..FontConfig::default() }),
    }]);

    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    let mut renderer =
        unsafe { imgui_dx11_renderer::Renderer::new(&mut imgui, device.clone()).unwrap() };
    let clear_color = [0.45, 0.55, 0.60, 1.00];

    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(_) => {
            let now = Instant::now();
            imgui.io_mut().update_delta_time(now - last_frame);
            last_frame = now;
        },
        Event::MainEventsCleared => {
            let io = imgui.io_mut();
            platform.prepare_frame(io, &window).expect("Failed to start frame");
            window.request_redraw();
        },
        Event::RedrawRequested(_) => {
            unsafe {
                context.OMSetRenderTargets(1, &main_rtv.as_raw(), ptr::null_mut());
                context.ClearRenderTargetView(main_rtv.as_raw(), &clear_color);
            }
            let ui = imgui.frame();
            imgui::Window::new(im_str!("Hello world"))
                .size([300.0, 100.0], imgui::Condition::FirstUseEver)
                .build(&ui, || {
                    ui.text(im_str!("Hello world!"));
                    ui.text(im_str!("This...is...imgui-rs!"));
                    ui.separator();
                    let mouse_pos = ui.io().mouse_pos;
                    ui.text(im_str!("Mouse Position: ({:.1},{:.1})", mouse_pos[0], mouse_pos[1]));
                });
            ui.show_demo_window(&mut true);
            platform.prepare_render(&ui, &window);
            renderer.render(ui.render()).unwrap();
            unsafe {
                swapchain.Present(1, 0);
            }
        },
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            *control_flow = winit::event_loop::ControlFlow::Exit
        },
        Event::WindowEvent {
            event: WindowEvent::Resized(winit::dpi::PhysicalSize { height, width }),
            ..
        } => unsafe {
            ptr::drop_in_place(&mut main_rtv);
            swapchain.ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
            ptr::write(&mut main_rtv, create_render_target(&swapchain, &device));
            platform.handle_event(imgui.io_mut(), &window, &event);
        },
        Event::LoopDestroyed => (),
        event => {
            platform.handle_event(imgui.io_mut(), &window, &event);
        },
    });
}
