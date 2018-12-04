#include <iostream>
#include <fmt/format.h>

#include "clara.hpp"
#include "graph/logger.hpp"
#include "graph/triangle_counter.hpp"
#include "graph/config.hpp"

int main(int argc, char **argv)
{

	Config config;

	std::string adjacencyListPath;
	bool help = false;
	bool debug = false;
	bool verbose = false;
	bool seedSet = false;

	clara::Parser cli;
	cli = cli | clara::Help(help);
	cli = cli | clara::Opt(debug)
					["--debug"]("print debug messages to stderr");
	cli = cli | clara::Opt(verbose)
					["--verbose"]("print verbose messages to stderr");
	cli = cli | clara::Opt(config.numCPUThreads_, "int")
					["-c"]["--num_cpu"]("number of cpu threads (default = automatic)");
	cli = cli | clara::Opt(config.gpus_, "ids")
					["-g"]("gpus to use");
	cli = cli | clara::Opt([&](unsigned int seed) {
			  seedSet = true;
			  config.seed_ = seed;
			  return clara::detail::ParserResult::ok(clara::detail::ParseResultType::Matched);
		  },
						   "int")["-s"]["--seed"]("random seed");
	cli = cli | clara::Opt(config.type_, "cpu|csr|cudamemcpy|nvgraph|um|zc")["-m"]["--method"]("method (default = um)").required();
	cli = cli | clara::Arg(adjacencyListPath, "graph file")("Path to adjacency list").required();

	auto result = cli.parse(clara::Args(argc, argv));
	if (!result)
	{
		LOG(error, "Error in command line: {}", result.errorMessage());
		exit(1);
	}

	if (help)
	{
		std::cout << cli;
		return 0;
	}

	if (verbose)
	{
		logger::console->set_level(spdlog::level::trace);
	}
	else if (debug)
	{
		logger::console->set_level(spdlog::level::debug);
	}
	if (seedSet)
	{
		LOG(debug, "using seed {}", config.seed_);
		srand(config.seed_);
	}
	else
	{
		uint seed = time(NULL);
		LOG(debug, "using seed {}", seed);
		srand(time(NULL));
	}

#ifndef TRI_RELEASE
	LOG(warn, "Not a release build");
#endif
	TriangleCounter *tc;
	tc = TriangleCounter::CreateTriangleCounter(config);

	auto start = std::chrono::system_clock::now();
	tc->read_data(adjacencyListPath);
	const size_t numEdges = tc->num_edges();
	double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
	LOG(info, "read_data time {}s", elapsed);

	start = std::chrono::system_clock::now();
	tc->setup_data();
	elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
	LOG(info, "setup_data time {}s", elapsed);

	start = std::chrono::system_clock::now();
	const auto numTriangles = tc->count();
	elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
	LOG(info, "count time {}s", elapsed);

	fmt::print("{} {} {} {}\n", adjacencyListPath, numTriangles, elapsed, numEdges / elapsed);

	delete tc;
	return 0;
}
