
#pragma once

namespace rpp {

template<typename T>
struct Is_Const {
    static constexpr bool value = false;
};

template<typename T>
struct Is_Const<const T> {
    static constexpr bool value = true;
};

template<typename T>
struct Add_Const {
    using type = const T;
};

template<typename T>
struct Remove_Const {
    using type = T;
};

template<typename T>
struct Remove_Const<const T> {
    using type = T;
};

template<typename T>
concept Const = Is_Const<T>::value;

template<typename T>
struct Is_Pointer {
    static constexpr bool value = false;
};

template<typename T>
struct Is_Pointer<T*> {
    static constexpr bool value = true;
};

template<typename T>
struct Add_Pointer {
    using type = T*;
};

template<typename T>
struct Remove_Pointer {
    using type = T;
};

template<typename T>
struct Remove_Pointer<T*> {
    using type = T;
};

template<typename T>
concept Pointer = Is_Pointer<T>::value;

template<typename T>
struct Is_Reference {
    static constexpr bool value = false;
};

template<typename T>
struct Is_Reference<T&> {
    static constexpr bool value = true;
};

template<typename T>
struct Is_Reference<T&&> {
    static constexpr bool value = true;
};

template<typename T>
struct Add_Reference {
    using type = T&;
};

template<typename T>
struct Remove_Reference {
    using type = T;
};

template<typename T>
struct Remove_Reference<T&> {
    using type = T;
};

template<typename T>
struct Remove_Reference<T&&> {
    using type = T;
};

template<typename T>
concept Reference = Is_Reference<T>::value;

template<typename T>
struct Decay {
    using type = T;
};

template<typename T>
struct Decay<const T> {
    using type = typename Decay<T>::type;
};

template<typename T>
struct Decay<T&> {
    using type = typename Decay<T>::type;
};

template<typename T>
struct Decay<T&&> {
    using type = typename Decay<T>::type;
};

template<bool b, typename T, typename F>
struct If;

template<typename T, typename F>
struct If<true, T, F> {
    using type = T;
};
template<typename T, typename F>
struct If<false, T, F> {
    using type = F;
};

template<typename L, typename R>
struct Is_Same {
    static constexpr bool value = false;
};
template<typename T>
struct Is_Same<T, T> {
    static constexpr bool value = true;
};

template<typename L, typename R>
concept Same = Is_Same<L, R>::value;

template<typename F, typename... Args>
using Invoke_Result = std::invoke_result_t<F, Args...>;

template<typename Q, typename T, typename... Ts>
struct All_Are {
    static constexpr bool value = Is_Same<Q, T>::value && All_Are<Q, Ts...>::value;
};
template<typename Q, typename T>
struct All_Are<Q, T> {
    static constexpr bool value = Is_Same<Q, T>::value;
};

template<typename Q, typename... Ts>
concept All = All_Are<Q, Ts...>::value;

template<typename Q, typename T, typename... Ts>
struct One_Is {
    static constexpr bool value = Is_Same<Q, T>::value || One_Is<Q, Ts...>::value;
};
template<typename Q, typename T>
struct One_Is<Q, T> {
    static constexpr bool value = Is_Same<Q, T>::value;
};

template<typename Q, typename T, typename... Ts>
concept One = One_Is<Q, T, Ts...>::value;

template<typename B, typename D>
concept Base_Of = std::is_base_of_v<B, D>;

template<typename D, typename B>
concept Derived_From = std::is_base_of_v<B, D>;

template<typename T>
concept Abstract = std::is_abstract_v<T>;

template<typename T>
concept Not_Abstract = !std::is_abstract_v<T>;

template<typename T, typename... Args>
concept Constructable = std::is_constructible_v<T, Args...>;

template<typename T>
concept Default_Constructable = std::is_default_constructible_v<T>;

template<typename T>
concept Copy_Constructable = std::is_copy_constructible_v<T>;

template<typename T>
concept Move_Constructable = std::is_move_constructible_v<T>;

template<typename T>
concept Trivially_Copyable = std::is_trivially_copyable_v<T>;

template<typename T>
concept Trivially_Movable = std::is_trivially_move_constructible_v<T>;

template<typename T>
concept Trivially_Destructible = std::is_trivially_destructible_v<T>;

template<typename T>
concept Must_Destruct = !std::is_trivially_destructible_v<T>;

template<typename F, typename... Args>
concept Invocable = std::is_invocable_v<F, Args...>;

template<typename T>
struct Is_Float {
    static constexpr bool value = false;
};

template<>
struct Is_Float<f32> {
    static constexpr bool value = true;
};

template<>
struct Is_Float<f64> {
    static constexpr bool value = true;
};

template<typename T>
concept Float = Is_Float<T>::value;

template<typename T>
struct Is_Int {
    static constexpr bool value = false;
};

template<>
struct Is_Int<u8> {
    static constexpr bool value = true;
};

template<>
struct Is_Int<i8> {
    static constexpr bool value = true;
};

template<>
struct Is_Int<u16> {
    static constexpr bool value = true;
};

template<>
struct Is_Int<i16> {
    static constexpr bool value = true;
};

template<>
struct Is_Int<u32> {
    static constexpr bool value = true;
};

template<>
struct Is_Int<i32> {
    static constexpr bool value = true;
};

template<>
struct Is_Int<u64> {
    static constexpr bool value = true;
};

template<>
struct Is_Int<i64> {
    static constexpr bool value = true;
};

template<typename I>
concept Int = Is_Int<I>::value;

template<typename T>
concept Equality = requires(T l, T r) {
    { l == r } -> Same<bool>;
    { l != r } -> Same<bool>;
};

template<u64 N, typename... Ts>
concept Length = sizeof...(Ts) == N;

template<typename T>
concept Trivial = std::is_trivial_v<T>;

template<typename T>
concept Clone = Move_Constructable<T> && requires(T value) {
    { value.clone() } -> Same<T>;
};

#define CASE(case_) Case<T, T::case_, #case_>
#define FIELD(field) Field<decltype(T::field), offsetof(T, field), #field>

namespace detail {

struct Literal {
    static constexpr u64 max_len = 32;
    template<size_t N>
    constexpr Literal(const char (&literal)[N]) {
        static_assert(N <= max_len);
        for(u64 i = 0; i < N; i++) {
            c_string[i] = literal[i];
        }
    }
    char c_string[max_len] = {};
};

namespace list {

struct Nil {};

template<typename H, typename T>
struct Cons {
    using Head = H;
    using Tail = T;
};

template<typename T>
struct Is_List {
    static constexpr bool value = false;
};

template<>
struct Is_List<Nil> {
    static constexpr bool value = true;
};

template<typename Head, typename Tail>
struct Is_List<Cons<Head, Tail>> {
    static constexpr bool value = Is_List<Tail>::value;
};

template<typename T>
concept List = Is_List<T>::value;

template<typename... Ts>
struct Make;

template<typename T, typename... Ts>
struct Make<T, Ts...> {
    using L = Cons<T, typename Make<Ts...>::L>;
};

template<>
struct Make<> {
    using L = Nil;
};

template<typename F, typename X>
concept Predicate = requires() {
    { F::template value<X> } -> Same<bool>;
};

template<typename F, typename X>
concept Apply = requires(F f) {
    { f.template apply<X>() } -> Same<void>;
};

template<typename F, List L>
struct All;

template<typename F>
struct All<F, Nil> {
    static constexpr bool value = true;
};

template<typename F, typename Head, List Tail>
    requires Predicate<F, Head>
struct All<F, Cons<Head, Tail>> {
    static constexpr bool value = F::template value<Head> && All<F, Tail>::value;
};

template<typename F, List L>
struct Iter;

template<typename F>
struct Iter<F, Nil> {
    static void apply(F&& f) {
    }
};

template<typename F, typename Head, List Tail>
    requires Apply<F, Head>
struct Iter<F, Cons<Head, Tail>> {
    static void apply(F&& f) {
        f.template apply<Head>();
        Iter<F, Tail>::apply(std::forward<F>(f));
    }
};

} // namespace list

template<typename... Ts>
using List = typename list::Make<Ts...>::L;

template<typename T>
concept Type_List = list::List<T>;

template<typename P, typename L>
concept Type_List_All = list::All<P, L>::value;

enum class Kind : u8 {
    void_,
    i8_,
    i16_,
    i32_,
    i64_,
    u8_,
    u16_,
    u32_,
    u64_,
    f32_,
    f64_,
    bool_,
    array_,
    pointer_,
    enum_,
    record_
};

template<typename T>
struct Reflect;

template<typename T>
concept Reflectable = requires() {
    { Reflect<T>::name } -> Same<Literal>;
    { Reflect<T>::kind } -> Same<Kind>;
};

template<typename E, E V, Literal N>
struct Case {
    using type = E;
    static constexpr Literal name = N;
    static constexpr E value = V;
};

template<typename F, u64 O, Literal N>
struct Field {
    using type = F;
    static constexpr Literal name = N;
    static constexpr u64 offset = O;
};

template<typename EC>
concept Enum_Case = requires() {
    Reflectable<typename EC::type>;
    { EC::name } -> Same<Literal>;
    { EC::value } -> Same<typename EC::type>;
};

struct Is_Enum_Case {
    template<typename EC>
    static constexpr bool value = Enum_Case<EC>;
};

template<typename RF>
concept Record_Field = requires() {
    Reflectable<typename RF::type>;
    { RF::name } -> Same<Literal>;
    { RF::offset } -> Same<u64>;
};

struct Is_Record_Field {
    template<typename RF>
    static constexpr bool value = Record_Field<RF>;
};

template<typename E>
concept Enum = requires() {
    Reflectable<E>;
    Reflectable<typename Reflect<E>::underlying>;
    Type_List_All<Is_Enum_Case, typename Reflect<E>::members>;
    Same<E, decltype(Reflect<E>::default_)>;
    Reflect<E>::kind == Kind::enum_;
};

template<typename R>
concept Record = requires() {
    Reflectable<R>;
    Type_List_All<Is_Record_Field, typename Reflect<R>::members>;
    Reflect<R>::kind == Kind::record_;
};

template<typename F>
struct Invoke_Case {
    template<typename C>
        requires Invocable<F, const Literal&, typename C::type>
    void apply() {
        f(C::name, C::value);
    }
    F f;
};

template<typename F, typename T>
concept Field_Iterator = requires(F f, const Literal& name, T& field) {
    { f.template apply<T>(name, field) } -> Same<void>;
};

template<typename F>
struct Invoke_Field {
    template<typename RF>
        requires Field_Iterator<F, typename RF::type>
    void apply() {
        using T = typename RF::type;
        T* field = reinterpret_cast<T*>(address + RF::offset);
        f.template apply<T>(RF::name, *field);
    }
    F f;
    u8* address;
};

template<Enum E, typename F>
    requires Invocable<F, const Literal&, E>
void iterate_enum(F&& f) {
    list::Iter<Invoke_Case<F>, typename Reflect<E>::members>::apply(
        Invoke_Case<F>{std::forward<F>(f)});
}

template<Record R, typename F>
void iterate_record(F&& f, R& record) {
    u8* address = reinterpret_cast<u8*>(&record);
    list::Iter<Invoke_Field<F>, typename Reflect<R>::members>::apply(
        Invoke_Field<F>{std::forward<F>(f), address});
}

template<typename T>
struct Reflect<const T> : Reflect<T> {};

template<typename T>
struct Reflect<T&> : Reflect<T> {};

template<typename T>
struct Reflect<T&&> : Reflect<T> {};

template<typename T>
struct Reflect<T*> {
    using underlying = T;
    static constexpr Literal name = Reflect<T>::name;
    static constexpr Kind kind = Kind::pointer_;
};

template<typename T, u64 N>
struct Reflect<T[N]> {
    using underlying = T;
    static constexpr Literal name = Reflect<T>::name;
    static constexpr u64 length = N;
    static constexpr Kind kind = Kind::array_;
};

template<>
struct Reflect<void> {
    static constexpr Literal name = "void";
    static constexpr Kind kind = Kind::void_;
};

template<>
struct Reflect<decltype(null)> {
    using underlying = void;
    static constexpr Literal name = "null";
    static constexpr Kind kind = Kind::pointer_;
};

template<>
struct Reflect<i8> {
    static constexpr Literal name = "i8";
    static constexpr Kind kind = Kind::i8_;
};

template<>
struct Reflect<u8> {
    static constexpr Literal name = "u8";
    static constexpr Kind kind = Kind::u8_;
};

template<>
struct Reflect<i16> {
    static constexpr Literal name = "i16";
    static constexpr Kind kind = Kind::i16_;
};

template<>
struct Reflect<u16> {
    static constexpr Literal name = "u16";
    static constexpr Kind kind = Kind::u16_;
};
template<>
struct Reflect<i32> {
    static constexpr Literal name = "i32";
    static constexpr Kind kind = Kind::i32_;
};

template<>
struct Reflect<u32> {
    static constexpr Literal name = "u32";
    static constexpr Kind kind = Kind::u32_;
};

template<>
struct Reflect<i64> {
    static constexpr Literal name = "i64";
    static constexpr Kind kind = Kind::i64_;
};

template<>
struct Reflect<u64> {
    static constexpr Literal name = "u64";
    static constexpr Kind kind = Kind::u64_;
};

template<>
struct Reflect<f32> {
    static constexpr Literal name = "f32";
    static constexpr Kind kind = Kind::f32_;
};

template<>
struct Reflect<f64> {
    static constexpr Literal name = "f64";
    static constexpr Kind kind = Kind::f64_;
};

template<>
struct Reflect<bool> {
    static constexpr Literal name = "bool";
    static constexpr Kind kind = Kind::bool_;
};

template<>
struct Reflect<Kind> {
    using T = Kind;
    using underlying = u8;
    static constexpr Literal name = "Kind";
    static constexpr Kind kind = Kind::enum_;
    static constexpr Kind default_ = Kind::void_;
    using members = List<CASE(void_), CASE(i8_), CASE(i16_), CASE(i32_), CASE(i64_), CASE(u8_),
                         CASE(u16_), CASE(u32_), CASE(u64_), CASE(f32_), CASE(f64_), CASE(bool_),
                         CASE(array_), CASE(pointer_), CASE(enum_), CASE(record_)>;
    static_assert(Enum<T>);
};

} // namespace detail

using detail::Enum;
using detail::Kind;
using detail::Literal;
using detail::Record;
using detail::Reflect;

using detail::iterate_enum;
using detail::iterate_record;

} // namespace rpp
