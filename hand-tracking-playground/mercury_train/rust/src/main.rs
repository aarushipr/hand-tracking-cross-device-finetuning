use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Sub, SubAssign};

use csv;
use stereokit::input::{ButtonState, StereoKitInput};
use stereokit::lifecycle::{DisplayMode, StereoKitDraw};
use stereokit::{lifecycle, model, time::StereoKitTime, ui, Settings, StereoKit};
use stereokit_sys;

use clap::Parser;
use stereokit::sys::{input_mouse, matrix_s};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
	#[arg(short, long)]
	euroc_path: String,
}

struct view {
	img_tex: stereokit::texture::Texture,
	img_mat: stereokit::material::Material,
	img_model: stereokit::model::Model,
}

struct State {
	euroc_path: String,

	entries: [Vec<EurocEntry>; 2],
	sequence_length: usize,
	current_idx: usize,


	global_mouse_position: glam::f32::Vec2,
    global_mouse_position_last_frame: glam::f32::Vec2,
	// this_frame_textures: [stereokit::texture::Texture; 2],
	views: [view; 2],
}

#[derive(Default)]
struct EurocEntry {
	timestamp: i64,
	full_path: String,
}



fn bleh(blah: String) -> Vec<EurocEntry> {
	let file = std::fs::File::open(blah.clone() + "/data.csv").unwrap();
	let reader = std::io::BufReader::new(file);
	let mut rdr = csv::Reader::from_reader(reader);

	let mut ees: Vec<EurocEntry> = Vec::new();

	for result in rdr.records() {
		// The iterator yields Result<StringRecord, Error>, so we check the
		// error here.
		let record = result.unwrap();

		let mut entry: EurocEntry = EurocEntry::default();
		entry.timestamp = record[0].parse().unwrap();
		entry.full_path = blah.clone() + "/data/" + &record[1];

		// println!("{:?}", record);
		// println!("{:?}", entry.timestamp);
		// println!("{:?}", entry.full_path);
		ees.push(entry);
	}

	ees
}

fn update_global_mouse(sk: &StereoKitDraw, state: &mut State) {
	state.global_mouse_position_last_frame = state.global_mouse_position;

    let mut bleh: stereokit::values::Ray = stereokit::values::Ray::from_mouse(&sk.input_mouse());


	state.global_mouse_position.x = bleh.pos.x;
	state.global_mouse_position.y = bleh.pos.y;

	bleh.pos.z -= 0.7;

	let mut blah = stereokit::pose::Pose::new(bleh.pos, glam::Quat::IDENTITY);

	// println!("{:?}", blah);

	stereokit::lines::line_add_axis(sk, blah, 0.01);


	// println!("{:?}", bleh.pos);
	// println!("{:?}", bleh.dir);
}

/*
void
manage_pan_zoom()
{
	const sk::mouse_t *mouse = input_mouse();
	if (mouse->scroll_change != 0.0) {
		sk::matrix root = sk::render_get_cam_root();
		float s = 0.88;
		if (mouse->scroll_change < 0.0) {
			s = 1 / s; // 1.08;
		}

		sk::matrix matrix_scale = sk::matrix_s({s, s, 1});
		sk::vec3 trans = sk::vec3{st.global_mouse_position.x, st.global_mouse_position.y, 0} * (1 - s);
		trans.z = 0;
		sk::matrix matrix_translate = sk::matrix_t(trans);
		sk::matrix transform;

		sk::matrix_mul(matrix_scale, matrix_translate, transform);

		sk::matrix new_;

		sk::matrix_mul(root, transform, new_);

		sk::render_set_cam_root(new_);

		// We just moved the camera, recalculate mouse position!
		update_global_mouse();
	}

	else if (sk::input_key(sk::key_mouse_center)) {
		sk::matrix root = sk::render_get_cam_root();
		ray_t old;
		ray_from_mouse(mouse->pos - mouse->pos_change, old);
		sk::vec2 move = sk::vec2{old.pos.x, old.pos.y} - st.global_mouse_position;
		sk::matrix new_;
		sk::matrix_mul(root, sk::matrix_trs({move.x, move.y, 0}), new_);
		sk::render_set_cam_root(new_);

		// We just moved the camera, recalculate mouse position!
		update_global_mouse();
	}
}

 */

fn manage_pan_zoom(sk: &stereokit::lifecycle::StereoKitDraw, state: &mut State) {
	let mouse = sk.input_mouse();


	if mouse.scroll_change != 0.0 {
		let root: glam::Mat4 = stereokit::render::Camera::get_root(sk).into();
		let mut s: f32 = 0.88f32;
		if mouse.scroll_change < 0.0f32 {
			s = 1.0f32 / s;
		}

		let mut matrix_scale: glam::Mat4 = glam::Mat4::from_scale(glam::Vec3 {
			x: s,
			y: s,
			z: 1.0f32,
		});

		let mut trans = glam::Vec3 {
			x: state.global_mouse_position.x,
			y: state.global_mouse_position.y,
			z: 0.0f32,
		};

        trans *= (1.0f32-s);

        let mut translate: glam::Mat4 = glam::Mat4::from_translation(trans);


        let mut transform: glam::Mat4 = translate * matrix_scale;

        let mut new_: glam::Mat4 = transform * root ;



        stereokit::render::Camera::set_root(sk, new_);
        update_global_mouse(sk, state);
    } else if stereokit::input::StereoKitInput::input_key(sk, stereokit::input::Key::MouseLeft) == ButtonState::Active {
        let root: glam::Mat4 = stereokit::render::Camera::get_root(sk).into();
        // let mut old: stereokit::values::Ray = fixed_from_mouse(mouse.pos - mouse.pos_change);

        // glam::vec2 old
         let mut move_: glam::Vec2 = state.global_mouse_position_last_frame - state.global_mouse_position;

        let mut move_matrix: glam::Mat4 = glam::Mat4::from_translation(glam::Vec3{x: move_.x, y: move_.y, z: 0.0f32});
        let mut new_: glam::Mat4 = move_matrix * root;


        stereokit::render::Camera::set_root(sk, new_);
        update_global_mouse(sk, state);


    }
}

fn update(sk: &stereokit::lifecycle::StereoKitDraw, state: &mut State) {
	// if state.this_frame_textures[0].is_none() {
	//   state.this_frame_textures = [Some(stereokit::texture::Texture::from_file(sk, state.entries[0][0].full_path.clone(), false, 0).unwrap()),
	//     Some(stereokit::texture::Texture::from_file(sk, state.entries[0][0].full_path.clone(), false, 0).unwrap())]
	// }

	let mut width: i32 = 0;
	let mut height: i32 = 0;

	// width = state.this_frame_textures[0].get_width();
	// height = state.this_frame_textures[0].get_height();

	state.views[0].img_model.draw(
		sk,
		glam::f32::Mat4::from_scale_rotation_translation(
			glam::f32::vec3(1.0, 1.0, 1.0),
			glam::f32::Quat::IDENTITY,
			glam::f32::vec3(0., 0., -1.0),
		)
		.into(),
		stereokit::color_named::WHITE,
		stereokit::render::RenderLayer::Layer0,
	);

	update_global_mouse(sk, state);
    manage_pan_zoom(sk, state);
}

// &impl StereoKitContext

fn make_view(sk_instance: &stereokit::StereoKit, path: String) -> view {
	println!("{}", path);

	let tex = stereokit::texture::Texture::from_file(sk_instance, path, false, 0).unwrap();
	let mesh = stereokit::mesh::Mesh::gen_plane(
		sk_instance,
		glam::Vec2 {
			x: 1.0f32,
			y: 1.0f32,
		},
		glam::vec3(0.0f32, 0.0f32, 1.0f32),
		glam::vec3(0.0f32, 1.0f32, 0.0f32),
		1,
	)
	.unwrap();

	// let mesh = stereokit::mesh::Mesh::gen_cube(sk_instance, )

	let material = stereokit::material::Material::copy_from_id(
		sk_instance,
		stereokit::material::DEFAULT_ID_MATERIAL_UNLIT,
	)
	.unwrap();

	material.set_parameter(sk_instance, "tex_scale", &1.0f32);
	material.set_cull(sk_instance, stereokit::material::Cull::None);
	material.set_parameter(sk_instance, "diffuse", &tex);

	let model = stereokit::model::Model::from_mesh(sk_instance, &mesh, &material).unwrap();

	view {
		img_mat: material,
		img_tex: tex,
		img_model: model,
	}
}

fn main() {
	let mut sk_settings = Settings::default();

	sk_settings = sk_settings.log_filter(lifecycle::LogFilter::None);
	sk_settings = sk_settings.app_name("annotator");
	sk_settings = sk_settings.display_preference(DisplayMode::Flatscreen);
	sk_settings = sk_settings.disable_flatscreen_mr_sim(true);

	let sk_instance: stereokit::StereoKit = sk_settings.init().unwrap();

	unsafe {
		stereokit::sys::render_set_projection(stereokit::sys::projection__projection_ortho);
		stereokit::sys::render_enable_skytex(stereokit::sys::bool32_t::from(false));
	}

	let mut args = Args::parse();

	let mut entries0 = bleh(args.euroc_path.clone() + "/mav0/cam0/");
	let mut entries1 = bleh(args.euroc_path.clone() + "/mav0/cam1/");

	let mut path0 = entries0[0].full_path.clone();
	let mut path1 = entries1[0].full_path.clone();

	let mut len: usize = entries0.len();

	// let mut views:

	let mut state = State {
		euroc_path: args.euroc_path.clone(),
		entries: [entries0, entries1],
		sequence_length: len,
		current_idx: 0,
        global_mouse_position: glam::f32::Vec2 {
            x: 0.0f32,
            y: 0.0f32,
        },
        global_mouse_position_last_frame: glam::f32::Vec2 {
            x: 0.0f32,
            y: 0.0f32,
        },
		views: [
			make_view(&sk_instance, path0),
			make_view(&sk_instance, path1),
		],
		// this_frame_textures: [stereokit::texture::Texture::from_file(&sk_instance, path0, false, 0).unwrap(),
		//   stereokit::texture::Texture::from_file(&sk_instance, path1, false, 0).unwrap(),
		// ],
	};

	sk_instance.run(
		|ctx| {
			update(ctx, &mut state);
		},
		|_| {},
	);
}
