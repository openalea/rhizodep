import numpy as np
from collections.abc import MutableMapping


class ArrayDict(MutableMapping):
    """
    Mapping[int -> float] backed by two aligned arrays:
      - order[i] = key at logical position i  (sorted ascending, invariant)
      - arr[i]   = value for order[i]
    Also keeps a dict key -> current index for O(1) lookups/updates.
    """

    def __init__(self, init=None, dtype=np.float64, init_capacity=0):
        cap = max(int(init_capacity), 16)
        self.arr   = np.empty(cap, dtype=dtype)
        self.order = np.empty(cap, dtype=np.int64)
        self.vid2idx = {}
        self.size = 0

        if init:
            # Insert via __setitem__ to preserve sorted invariant
            for k, v in init.items():
                self[k] = v

    # --- capacity management -------------------------------------------------

    def _ensure(self, need):
        if need <= self.arr.size:
            return
        new = max(int(need), 2 * int(self.arr.size))
        new_arr   = np.empty(new, dtype=self.arr.dtype)
        new_order = np.empty(new, dtype=self.order.dtype)
        # copy current slice
        if self.size:
            new_arr[:self.size]   = self.arr[:self.size]
            new_order[:self.size] = self.order[:self.size]
        self.arr, self.order = new_arr, new_order

    # --- basic mapping protocol ---------------------------------------------

    def __getitem__(self, k: int) -> float:
        return float(self.arr[self.vid2idx[k]])

    def __len__(self):  # mapping size
        return self.size

    def __iter__(self):  # iterate keys in sorted order
        # Cast to int to avoid numpy scalar types leaking out
        for i in range(self.size):
            yield int(self.order[i])

    # --- sorted insert / delete ---------------------------------------------

    def __setitem__(self, k: int, v: float):
        idx = self.vid2idx.get(k)
        if idx is not None:  # existing -> O(1) update
            self.arr[idx] = v
            return

        self._ensure(self.size + 1)
        pos = int(np.searchsorted(self.order[:self.size], k))  # keep ascending

        # shift right suffix [pos:size)
        if pos < self.size:
            self.arr[pos+1:self.size+1]   = self.arr[pos:self.size]
            self.order[pos+1:self.size+1] = self.order[pos:self.size]

        # insert
        self.order[pos] = k
        self.arr[pos]   = v
        self.size += 1

        # rebuild mapping for moved suffix (including new key)
        for i in range(pos, self.size):
            self.vid2idx[int(self.order[i])] = i


    def __delitem__(self, k: int):
        idx = self.vid2idx.pop(k)  # KeyError if absent

        if idx < self.size - 1:
            # shift left suffix (idx+1:size)
            self.arr[idx:self.size-1]   = self.arr[idx+1:self.size]
            self.order[idx:self.size-1] = self.order[idx+1:self.size]

        self.size -= 1

        # rebuild mapping for moved suffix
        for i in range(idx, self.size):
            self.vid2idx[int(self.order[i])] = i


    # --- handy array views ---------------------------------------------------

    def values_array(self) -> np.ndarray:
        """Values aligned with keys in ascending key order."""
        return self.arr[:self.size]

    def keys_array(self) -> np.ndarray:
        """Keys (ascending)."""
        return self.order[:self.size].copy()

    # --- indexed / scatter ops ----------------------------------------------

    def indices_of(self, vids):
        return np.fromiter((self.vid2idx[v] for v in vids),
                           count=len(vids), dtype=np.int64)

    def assign_all(self, values):
        values = np.asarray(values, dtype=self.arr.dtype)
        if values.shape[0] != self.size:
            raise ValueError(f"assign_all length mismatch: got {values.shape[0]}, need {self.size}")
        self.arr[:self.size] = values

    def assign_at(self, idxs, values):
        self.arr[np.asarray(idxs, np.int64)] = np.asarray(values, dtype=self.arr.dtype)

    def scatter(self, keys, values):
        self.assign_at(self.indices_of(keys), values)

    # --- batch update that preserves sorting --------------------------------

    def update(self, d: dict):
        if not d:
            return

        has = self.vid2idx.__contains__
        existing, new_items = [], []
        for k, v in d.items():
            (existing if has(k) else new_items).append((k, v))

        # 1) existing keys -> scatter in place
        if existing:
            self.scatter([k for k, _ in existing], [v for _, v in existing])

        # 2) new keys -> keep array sorted
        if not new_items:
            return

        new_items.sort(key=lambda kv: kv[0])  # sort by key
        nk = np.fromiter((k for k, _ in new_items), dtype=np.int64, count=len(new_items))
        nv = np.asarray([v for _, v in new_items], dtype=self.arr.dtype)

        # fast append if monotone extension
        if self.size == 0 or nk[0] >= int(self.order[self.size - 1]):
            self._ensure(self.size + nk.size)
            start, end = self.size, self.size + nk.size
            self.order[start:end] = nk
            self.arr[start:end]   = nv
            for i in range(start, end):  # rebuild mapping for appended
                self.vid2idx[int(self.order[i])] = i
            self.size = end
            return

        # otherwise do a full merge
        ok = self.order[:self.size].copy()
        ov = self.arr[:self.size].copy()

        total = ok.size + nk.size
        self._ensure(total)

        i = j = t = 0
        while i < ok.size and j < nk.size:
            if ok[i] <= nk[j]:
                self.order[t] = ok[i]; self.arr[t] = ov[i]; i += 1
            else:
                self.order[t] = nk[j]; self.arr[t] = nv[j]; j += 1
            t += 1

        if i < ok.size:
            r = ok.size - i
            self.order[t:t+r] = ok[i:]; self.arr[t:t+r] = ov[i:]; t += r
        if j < nk.size:
            r = nk.size - j
            self.order[t:t+r] = nk[j:]; self.arr[t:t+r] = nv[j:]; t += r

        self.size = t
        # rebuild full mapping
        self.vid2idx.clear()
        for i in range(self.size):
            self.vid2idx[int(self.order[i])] = i



    # --- utilities -----------------------------------------------------------

    def to_dict(self):
        # Already in key order; order of dict doesn’t matter here
        return {int(k): float(v) for k, v in self.items()}
    
    def reindex_sorted_inplace(self):
        if self.size <= 1: return
        p = np.argsort(self.order[:self.size], kind="mergesort")
        self.order[:self.size] = self.order[:self.size][p]
        self.arr[:self.size]   = self.arr[:self.size][p]
        self.vid2idx.clear()
        for i, k in enumerate(self.order[:self.size]):
            self.vid2idx[int(k)] = i

    def check_invariant(self):
        if self.size == 0: return True
        keys = self.order[:self.size]
        if not np.all(keys[:-1] <= keys[1:]):
            return False
        
        for i, k in enumerate(keys):
            if not self.vid2idx[int(k)] == i:
                return False
        return True


def mtg_to_arraydict(g, ignore: list = []):
    props = g.properties()
    for k, v in props.items():
        # print(k, v)
        if isinstance(v, dict) and len(v) > 0 and k not in ignore:
            assigned_values = [value for value in v.values() if value is not None]
            if len(assigned_values) > 0:
                first_element = assigned_values[0]
                if isinstance(first_element, float) or isinstance(first_element, int) or isinstance(first_element, np.int32) or isinstance(first_element, np.int64) or isinstance(first_element, np.float64):
                    props[k] = ArrayDict(v)
        
        # If any was already existing, recreate it to make sure this is the right version with the invariant vid ordering # TODO remove after ArrayDict is stable
        elif isinstance(v, ArrayDict):
            stored = v.to_dict()
            props[k] = ArrayDict(stored)
                