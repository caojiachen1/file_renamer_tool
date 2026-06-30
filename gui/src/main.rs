mod app;
mod theme;

use iced::window;

fn main() -> iced::Result {
    iced::application(app::new, app::update, app::view)
        .theme(app::theme)
        .subscription(app::subscription)
        .window(window::Settings {
            size: iced::Size::new(1200.0, 800.0),
            maximized: true,
            ..Default::default()
        })
        .run()
}
