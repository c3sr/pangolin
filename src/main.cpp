#include <iostream>
#include <fmt/format.h>

#include "clara.hpp"
#include "graph/config.hpp"
#include "graph/configure.hpp"
#include "graph/logger.hpp"
#include "graph/triangle_counter/triangle_counter.hpp"
#include "graph/reader/edge_list_reader.hpp"

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
					["-c"]["--num_cpu"]("number of cpu threads");
	cli = cli | clara::Opt(config.gpus_, "ids")
					["-g"]("gpus to use");
	cli = cli | clara::Opt(config.hints_)
					["--unified-memory-hints"]("use unified memory hints");
	cli = cli | clara::Opt(config.storage_, "zc|um")
					["-s"]("GPU memory kind");
	cli = cli | clara::Opt([&](unsigned int seed) {
			  seedSet = true;
			  config.seed_ = seed;
			  return clara::detail::ParserResult::ok(clara::detail::ParseResultType::Matched);
		  },
						   "int")["--seed"]("random seed");
	cli = cli | clara::Opt(config.type_, "cpu|csr|cudamemcpy|edge|hu|impact|nvgraph|vertex")["-m"]["--method"]("method").required();
	cli = cli | clara::Opt(config.kernel_, "string")["-k"]["--kernel"]("kernel");
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

	// set logging level
	if (verbose)
	{
		logger::console->set_level(spdlog::level::trace);
	}
	else if (debug)
	{
		logger::console->set_level(spdlog::level::debug);
	}

	// log command line before much else happens
	{
		std::string cmd;
		for (int i = 0; i < argc; ++i)
		{
			if (i != 0)
			{
				cmd += " ";
			}
			cmd += argv[i];
		}
		LOG(debug, cmd);
	}
	LOG(debug, "graph version: {}.{}.{}", GRAPH_VERSION_MAJOR, GRAPH_VERSION_MINOR, GRAPH_VERSION_PATCH);
	LOG(debug, "graph branch:  {}", GRAPH_GIT_REFSPEC);
	LOG(debug, "graph sha:     {}", GRAPH_GIT_HASH);
	LOG(debug, "graph changes: {}", GRAPH_GIT_LOCAL_CHANGES);

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

#ifndef NDEBUG
	LOG(warn, "Not a release build");
#endif

	graph::EdgeListReader *reader;
	reader = graph::EdgeListReader::from_file(adjacencyListPath);
	LOG(trace, "created reader");

	{
		auto start = std::chrono::system_clock::now();
		EdgeList test = reader->read();
		double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
		LOG(info, "read_data time {}s", elapsed);
		LOG(info, "got {} edges", test.size());
	}

	TriangleCounter *tc;
	tc = TriangleCounter::CreateTriangleCounter(config);
	LOG(trace, "created triangle counter");

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
	auto numTriangles = tc->count();
	elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
	LOG(info, "count time {}s", elapsed);

	fmt::print("{} {} {} {}\n", adjacencyListPath, numTriangles, elapsed, numEdges / elapsed);

	delete tc;
	return 0;
}
