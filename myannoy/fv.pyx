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
    
    def PushBack(self, feature):
        self._thisptr.PushBack(feature)

    cpdef double Norm(self):
        return self._thisptr.Norm()
