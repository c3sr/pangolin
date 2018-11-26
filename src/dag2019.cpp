#include "graph/dag_2019.hpp"

DAG2019 DAG2019::from_edgelist(const EdgeList &l)
    {
        DAG2019 dag;

        // sort the edge list by src
        std::sort(l.begin(), l.end(), [](const Edge &a, const Edge &b) -> bool {
            return a.src < b.src;
        });


        for (const auto edge : l) {

            if (dag.nodes.empty() || dag.nodes.end()->src != edge.src) {
                dag.nodes.push_back(dag.edgeSrc.size());
            }

            dag.edgeSrc.push_back(e.src);
            dag.edgeDst.push_back(e.dst);
        }

        dag.nodes.push_back(dag.edgeSrc.size());
    }