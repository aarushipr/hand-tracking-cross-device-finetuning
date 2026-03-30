#include "stereokit.h"
#include "shls/CLI11.hpp"
#include <iostream>
#include "util/u_logging.h"

using namespace sk;
namespace fs = std::filesystem;

struct state
{
	fs::path euroc;
};

void
update(void *ptr)
{
	struct state *state_ptr = (state *)ptr;
	struct state &state_ref = *state_ptr;

  U_LOG_E("Meow");
}

int
main(int argc, char **argv)
{
	CLI::App app{"Bounding box annotator!!"};

	std::string euroc_path;
	app.add_option("--euroc_path", euroc_path, "Path to EuRoC dataset")->required(true);

	CLI11_PARSE(app, argc, argv);

	std::cout << euroc_path << std::endl;

	// st.paths.root = euroc_path;

	// cJSON *machine = open_file((st.paths.root / st.paths.machine_annotated).c_str());
	// cJSON *human = open_file((st.paths.root / st.paths.human_annotated).c_str());

	// st.json_root = human;
	// if (st.json_root == NULL) {
	// 	st.json_root = machine;
	// 	if (st.json_root == NULL) {
	// 		U_LOG_E(
	// 		    "EuRoC dataset does not contain a `machine_annotated.json` or a `human_annotated.json`!");
	// 		U_LOG_E("This is probably just a dataset that you never ran the auto-annotator script on.");
	// 		U_LOG_E("Run `py/run_machine_annotation.py --euroc_path %s`, then come back!",
	// 		        euroc_path.c_str());
	// 		exit(1);
	// 	}
	// }


	sk_settings_t settings = {};
	settings.app_name = "bounding box annotator!";
	settings.display_preference = sk::display_mode_flatscreen;
	settings.disable_flatscreen_mr_sim = true;
	if (!sk_init(settings))
		return 1;

	state *ours = new state;


	sk_run_data(update, ours, NULL, NULL);

	delete ours;

	return 0;
}