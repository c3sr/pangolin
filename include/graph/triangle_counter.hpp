#pragma once

class TriangleCounter {

    public:
    virtual void execute(const char *filename, int omp_numthreads);

};