# rpp

[![Windows](https://github.com/TheNumbat/rpp/actions/workflows/windows.yml/badge.svg?branch=main)](https://github.com/TheNumbat/rpp/actions/workflows/windows.yml)
[![Linux](https://github.com/TheNumbat/rpp/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/TheNumbat/rpp/actions/workflows/ubuntu.yml)

Minimal Rust-inspired C++20 STL replacement.
Refer to the [blog post](https://thenumb.at/rpp/) for details.

## Integration

To use rpp in your project, run the following command (or manually download the source):

```bash
git submodule add https://github.com/TheNumbat/rpp
```

Then add the following lines to your CMakeLists.txt:

```cmake
add_subdirectory(rpp)
target_link_libraries($your_target PRIVATE rpp)
target_include_directories($your_target PRIVATE "rpp")
```

To use rpp with another build system, add `rpp` to your include path, add `rpp/rpp/impl/unify.cpp` to the build, and add either `rpp/rpp/pos/unify.cpp` or `rpp/rpp/w32/unify.cpp` based on your platform.

## Build and Run Tests

To build rpp and run the tests, run the following commands:

```bash
mkdir build
cd build
cmake ..
cmake --build .
ctest
```

For faster parallel builds, you can instead generate [ninja](https://ninja-build.org/) build files with `cmake -G Ninja ..`.

## Platform Support

Only the following configurations are supported:

| OS      | Compiler    | Arch |
|---------|-------------|------|
| Windows | MSVC 19.37+ | AVX2 |
| Linux   | Clang 17+   | AVX2 |

Other configurations (macOS, aarch64, GCC, etc.) may be added in the future.

## Examples

### Logging

```cpp
#include <rpp/base.h>

i32 main() {
    assert(true);
    info("Information");
    warn("Warning");
    die("Fatal error (exits)");
}
```

### Data Structures

```cpp
#include <rpp/base.h>
#include <rpp/rc.h>
#include <rpp/stack.h>
#include <rpp/heap.h>
#include <rpp/tuple.h>
#include <rpp/variant.h>

using namespace rpp;

i32 main() {
    Ref<i32> ref;
    Box<i32, A> box;
    Rc<i32, A> rc;
    Arc<i32, A> arc;
    Opt<i32> optional;
    Storage<i32> storage;
    String<A> string;
    String_View string_view;
    Array<i32, 1> array;
    Vec<i32, A> vec;
    Slice<i32, A> slice;
    Stack<i32, A> stack;
    Queue<i32, A> queue;
    Heap<i32, A> heap;
    Map<i32, i32> map;
    Pair<i32, i32> pair;
    Tuple<i32, i32, i32> tuple;
    Variant<i32, f32> variant;
    Function<i32()> function;
}
```

### Allocators

```cpp
#include <rpp/base.h>

using namespace rpp;

i32 main() {
    using A = Mallocator<"A">;
    using B = Mallocator<"B">;
    {
        Vec<i32, A> a;
        Vec<i32, B> b;
        info("A allocated: %", a);
        info("B allocated: %", b);

        Box<i32, Mpool> pool;
        info("Pool allocated: %", pool);

        Region(R) {
            Vec<i32, Mregion<R>> region{1, 2, 3};
            info("Region allocated: %", region);
        }
    }
    Profile::finalize(); // Print statistics and check for leaks
}
```

### Reflection

```cpp
#include <rpp/base.h>

using namespace rpp;

struct Foo {
    i32 x;
    Vec<i32> y;
};
RPP_RECORD(Foo, RPP_FIELD(x), RPP_FIELD(y));

template<Reflectable T>
struct Bar {
    T t;
};
template<Reflectable T>
RPP_TEMPLATE_RECORD(Bar, T, RPP_FIELD(t));

i32 main() {
    Bar<Foo> bar{Foo{42, Vec<i32>{1, 2, 3}}};
    info("bar: %", bar);
}
```

### Async

```cpp
#include <rpp/base.h>
#include <rpp/pool.h>
#include <rpp/asyncio.h>

using namespace rpp;

i32 main() {
    Async::Pool<> pool;

    auto coro = [](Async::Pool<>& pool) -> Async::Task<i32> {
        co_await pool.suspend();
        info("Hello from thread %!", Thread::this_id());
        co_await AsyncIO::wait(pool, 100);
        co_return 0;
    };

    auto task = coro(pool);
    info("Task returned: %", task.block());
}
```

### Math

```cpp
#include <rpp/base.h>
#include <rpp/vmath.h>
#include <rpp/simd.h>

using namespace rpp;

i32 main() {
    Vec3 v{0.0f, 1.0f, 0.0f};
    Mat4 m = Mat4::translate(v);
    info("Translated: %", m * v);

    F32x8 simd = F32x8::set1(1.0f);
    info("Dot product: %", F32x8::dot(simd, simd));
}
```

## To-Dos

- Modules
- Async
    - [ ] scheduler priorities
    - [ ] scheduler affinity
    - [ ] scheduler work stealing
    - [ ] io_uring for Linux file IO
    - [ ] sockets
- Types
    - [ ] Result<T,E>
    - [ ] Map: don't store hashes of integer keys
- Allocators
    - [ ] Per-thread pools
- Misc
    - [ ] Range_Allocator: add second level of linear buckets
    - [ ] Range_Allocator: reduce overhead
