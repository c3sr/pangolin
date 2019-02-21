#pragma once

#ifdef PANGOLIN_BEGIN_NAMESPACE
#error PANGOLIN_BEGIN_NAMESPACE already defined
#endif

#define PANGOLIN_BEGIN_NAMESPACE() namespace pangolin {
#define PANGOLIN_END_NAMESPACE() } // namespce pangolin