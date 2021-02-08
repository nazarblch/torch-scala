#include <array>
template<typename P, std::size_t N = 0> class StdArrayAdapter: public std::array<P,N> {
public:
    StdArrayAdapter(const P* ptr, int size, void* owner) : 
        ptr((P*)ptr), 
        size(size), 
        owner(owner),
        vec2(ptr ? std::array<P,N>((P*)ptr, (P*)ptr + size) : std::array<P,N>()),
        vec(vec2) { }
    StdArrayAdapter(const std::array<P,N>& vec) : ptr(0), size(0), owner(0), vec2(vec), vec(vec2) { }
    StdArrayAdapter(      std::array<P,N >& vec) : ptr(0), size(0), owner(0), vec(vec) { }
    StdArrayAdapter(const std::array<P,N>* vec) : ptr(0), size(0), owner(0), vec(*(std::array<P,N>*)vec) { }
    void assign(P* ptr, int size, void* owner) {
        this->ptr = ptr;
        this->size = size;
        this->owner = owner;
        vec.assign(ptr, ptr + size);
    }
    static void deallocate(void* owner) { operator delete(owner); }
    operator P*() {
        if (vec.size() > size) {
            ptr = (P*)(operator new(sizeof(P) * vec.size(), std::nothrow_t()));
        }
        if (ptr) {
            std::copy(vec.begin(), vec.end(), ptr);
        }
        size = vec.size();
        owner = ptr;
        return ptr;
    }
    operator const P*()        { return &vec[0]; }
    operator std::array<P,N>&() { return vec; }
    operator std::array<P,N>*() { return ptr ? &vec : 0; }
    P* ptr;
    int size;
    void* owner;
    std::array<P,N> vec2;
    std::array<P,N>& vec;
};
