# distutils: language=c++
# distutils: sources=feature_vector.cpp
# distutils: extra_compile_args= -std=c++11 -fopenmp

cdef extern from "feature_vector.cpp":
    cdef cppclass FeatureVector:
        FeatureVector()
        void PushBack(double feature)
        double Norm()
        int Size()
        double At(int index)
    
cdef class FeatureVec:
    cdef FeatureVector* _thisptr

    def __cinit__(self):
        self._thisptr = new FeatureVector()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr
    
    cpdef append(self, feature):
        self._thisptr.PushBack(feature)

    cpdef double norm(self):
        return self._thisptr.Norm()

    cdef double at(self, int index):
        return self._thisptr.At(index)

    def __getitem__(self, int index):
        return self.at(index)

    cdef size(self):
        return self._thisptr.Size()

    def __len__(self):
        return self.size()

    def __str__(self):
        vals = [str(self[i]) for i in range(len(self))]
        return ' '.join(vals)
            
    cdef _dot(self, other):
        FeatureVector* newval = self._thisptr.dot(other._thisptr)
        if self._thisptr != NULL:
            del self._thisptr
        self._thisptr = newval

    def dot(self, other):
        self._dot(other)
