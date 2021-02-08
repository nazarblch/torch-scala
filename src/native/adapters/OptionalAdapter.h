#include "torch/all.h"

template<class T> class OptionalAdapter: public c10::optional<T> {
public:
    OptionalAdapter(const T* ptr, int size, void *owner) :
        ptr((T*)ptr),
        size(size),
        owner(owner),
        smartPtr2(owner != NULL && owner != ptr ? *(c10::optional<T>*)owner : c10::optional<T>(*(T*)ptr)),
        smartPtr(smartPtr2) { }
    OptionalAdapter(const c10::optional<T>& smartPtr) :
        ptr(0),
        size(0),
        owner(0),
        smartPtr2(smartPtr),
        smartPtr(smartPtr2) { }
    void assign(T* ptr, int size, void* owner) {
        this->ptr = ptr;
        this->size = size;
        this->owner = owner;
        this->smartPtr = owner != NULL && owner != ptr ? *(c10::optional<T>*)owner : c10::optional<T>(*(T*)ptr);
    }
    static void deallocate(void* owner) {
        delete (c10::optional<T>*)owner;
    }
    operator T*() {
        ptr = &smartPtr.value();
        if (owner == NULL || owner == ptr) {
            owner = new c10::optional<T>(smartPtr);
        }
        return ptr;
    }
    operator c10::optional<T>&() {
        return smartPtr;
    }
    operator c10::optional<T>*() {
        return ptr ? &smartPtr : 0;
    }
    T* ptr;
    int size;
    void* owner;
    c10::optional<T> smartPtr2;
    c10::optional<T>& smartPtr;
};