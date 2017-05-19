import numpy as np
cimport numpy as np

cpdef rainflow(np.ndarray extrema):
    cdef list residuen_vector = []

    cdef int rv_index
    cdef double first
    cdef double second
    cdef double third

    max_cycle_count = extrema.size / 2

    # table_obj = hdf5_file.table_cache[table]
    cdef np.ndarray cycles = np.zeros((int(max_cycle_count),2), dtype=np.float64)

    cdef int len_residuen_vector = 0
    cdef int array_index = 0
    for current_extremum in extrema:
        residuen_vector.append(current_extremum)
        len_residuen_vector += 1

        if len_residuen_vector < 3:
            continue

        rv_index = 0
        while rv_index + 2 < len_residuen_vector:
            first = residuen_vector[rv_index]
            second = residuen_vector[rv_index + 1]
            third = residuen_vector[rv_index + 2]

            if (first > second and third >= first) or (first < second and third <= first):
                value = (first, second)
                # table_obj.append(value)
                cycles[array_index] = value
                array_index += 1
                residuen_vector.pop(rv_index + 1)
                residuen_vector.pop(rv_index)
                rv_index = 0
                len_residuen_vector -= 2
            else:
                rv_index += 1

    return cycles[0:array_index], residuen_vector