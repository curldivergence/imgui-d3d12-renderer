use rusty_d3d12::*;
use std::{env, fs, ptr, slice, str};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    static VERTEX_SHADER_FILE_NAME: &str = "vertex_shader.hlsl";
    static VERTEX_SHADER_SOURCE: &str = include_str!("src/vertex_shader.hlsl");

    match hassle_rs::utils::compile_hlsl(
        "ImGuiVs",
        VERTEX_SHADER_SOURCE,
        "main",
        "vs_6_0",
        // &["/Zi", "/Zss", "/Od"],
        &[],
        &[],
    ) {
        Ok(bytecode) => {
            println!("Shader {} compiled successfully", VERTEX_SHADER_FILE_NAME);
            fs::write(&format!("{}/{}", out_dir, VERTEX_SHADER_FILE_NAME), &bytecode)
                .unwrap_or_else(|_| panic!("Unable to write shader {} to out dir", VERTEX_SHADER_FILE_NAME))
        }
        Err(error) => {
            panic!(
                "Error: {}: {}",
                error,
                DxError::new("compile_hlsl", (0x80004005 % u32::MAX) as i32)
            ); // E_FAIL
        }
    }

    static PIXEL_SHADER_FILE_NAME: &str = "pixel_shader.hlsl";
    static PIXEL_SHADER_SOURCE: &str = include_str!("src/pixel_shader.hlsl");

    match hassle_rs::utils::compile_hlsl(
        "ImGuiPs",
        PIXEL_SHADER_SOURCE,
        "main",
        "ps_6_0",
        // &["/Zi", "/Zss", "/Od"],
        &[],
        &[],
    ) {
        Ok(bytecode) => {
            println!("Shader {} compiled successfully", VERTEX_SHADER_FILE_NAME);
            fs::write(&format!("{}/{}", out_dir, PIXEL_SHADER_FILE_NAME), &bytecode)
                .unwrap_or_else(|_| panic!("Unable to write shader {} to out dir", PIXEL_SHADER_FILE_NAME))
        }
        Err(error) => {
            panic!(
                "Error: {}: {}",
                error,
                DxError::new("compile_hlsl", (0x80004005 % u32::MAX) as i32)
            ); // E_FAIL
        }
    }
}