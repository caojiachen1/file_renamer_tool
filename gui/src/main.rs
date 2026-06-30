mod app;
mod theme;

fn main() -> iced::Result {
    iced::application(app::new, app::update, app::view)
        .theme(app::theme)
        .subscription(app::subscription)
        .window_size((1200.0, 800.0))
        .run()
}
