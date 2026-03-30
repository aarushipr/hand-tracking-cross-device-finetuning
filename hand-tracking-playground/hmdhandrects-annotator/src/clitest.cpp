// #include "math/m_api.h"
// #include "some_defs.hpp"
// #include "2d_viz.hpp"
// #include "../monado/src/xrt/drivers/ht/templates/NaivePermutationSort.hpp"
#include "CLI11.hpp"
#include <iostream>

int
main(int argc, char** argv)
{
	CLI::App app{"Bounding box annotator!!"};

    std::string euroc_path = "default";
    app.add_option("--euroc_path", euroc_path, "A help string")->required(true);

    CLI11_PARSE(app, argc, argv);

    std::cout << euroc_path << std::endl;
    return 0;

}
