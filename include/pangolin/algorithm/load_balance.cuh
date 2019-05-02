#pragma once

/*!

  Some number of objects, each of which produces `count` work-items.
  Upper-bound search of work item index into exclusive scan of work-item counts.

*/
template <typename T, typename U>
inline void load_balance(T *indices,                //<! [out] the object index that produced each work item
                         const size_t numWorkItems, //<! [in] the total number of work items
                         const U *counts,           //<! [in] the number of work items produced by each object
                         const size_t numObjects    //<! [in] the number of objects

) {
  size_t wi = 0;
  T oi = 0;
  U exclScanCounts = 0;
  while (wi < numWorkItems || oi < numObjects) {
    bool pred;
    if (oi >= numObjects) { // all remaining work-items are from the last object
      pred = true;
    } else if (wi >= numWorkItems) { // all work items have been produced, final objects didn't contribute any
      pred = false;
    } else {
      pred = (wi < exclScanCounts);
    }
    if (pred) {
      indices[wi++] = oi - 1;
    } else {
      exclScanCounts += counts[oi];
      ++oi;
    }
  }
}