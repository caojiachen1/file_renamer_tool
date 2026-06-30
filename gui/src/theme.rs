use iced::theme::palette::Palette;
use iced::{Color, Theme};

/// Fluent Design dark theme
pub fn fluent_dark() -> Theme {
    Theme::custom(
        "Fluent Dark",
        Palette {
            background: Color::from_rgb8(32, 32, 32),        // #202020
            text: Color::from_rgb8(255, 255, 255),           // #FFFFFF
            primary: Color::from_rgb8(0, 120, 212),          // #0078D4 (Fluent accent)
            success: Color::from_rgb8(108, 201, 114),        // #6CC972
            warning: Color::from_rgb8(255, 185, 0),          // #FFB900
            danger: Color::from_rgb8(255, 99, 71),           // #FF6347
        },
    )
}

/// Fluent Design color constants
pub mod colors {
    use iced::Color;

    // Backgrounds
    pub const BG: Color = Color::from_rgb8(32, 32, 32);          // #202020
    pub const SURFACE: Color = Color::from_rgb8(44, 44, 44);     // #2C2C2C
    pub const CARD: Color = Color::from_rgb8(50, 50, 50);        // #323232
    pub const ELEVATED: Color = Color::from_rgb8(58, 58, 58);    // #3A3A3A

    // Borders
    pub const BORDER: Color = Color::from_rgb8(64, 64, 64);      // #404040
    pub const BORDER_SUBTLE: Color = Color::from_rgb8(80, 80, 80); // #505050

    // Text
    pub const TEXT: Color = Color::from_rgb8(255, 255, 255);      // #FFFFFF
    pub const TEXT_SECONDARY: Color = Color::from_rgb8(180, 180, 180); // #B4B4B4
    pub const TEXT_DISABLED: Color = Color::from_rgb8(120, 120, 120); // #787878

    // Fluent accent
    pub const ACCENT: Color = Color::from_rgb8(0, 120, 212);     // #0078D4
    pub const ACCENT_LIGHT: Color = Color::from_rgb8(38, 137, 218); // #2689DA
    pub const ACCENT_DARK: Color = Color::from_rgb8(0, 99, 177); // #0063B1

    // Button colors
    pub const BTN_PRIMARY: Color = Color::from_rgb8(0, 120, 212); // #0078D4
    pub const BTN_PRIMARY_HOVER: Color = Color::from_rgb8(16, 137, 226); // #1089E2
    pub const BTN_SECONDARY: Color = Color::from_rgb8(50, 50, 50); // #323232
    pub const BTN_SECONDARY_HOVER: Color = Color::from_rgb8(68, 68, 68); // #444444
    pub const BTN_DANGER: Color = Color::from_rgb8(196, 43, 28); // #C42B1C
    pub const BTN_DANGER_HOVER: Color = Color::from_rgb8(218, 59, 43); // #DA3B2B
    pub const BTN_SUCCESS: Color = Color::from_rgb8(108, 201, 114); // #6CC972
    pub const BTN_SUCCESS_HOVER: Color = Color::from_rgb8(126, 213, 131); // #7ED583

    // Input
    pub const INPUT_BG: Color = Color::from_rgb8(44, 44, 44);     // #2C2C2C
    pub const INPUT_BORDER: Color = Color::from_rgb8(80, 80, 80); // #505050
    pub const INPUT_FOCUS: Color = Color::from_rgb8(0, 120, 212); // #0078D4
    pub const INPUT_PLACEHOLDER: Color = Color::from_rgb8(120, 120, 120); // #787878
}
