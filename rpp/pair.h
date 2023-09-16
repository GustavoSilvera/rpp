
#pragma once

namespace rpp {

template<typename A, typename B>
struct Pair {

    Pair()
        requires Default_Constructable<A> && Default_Constructable<B>
    = default;

    explicit Pair(const A& first, const B& second)
        requires Trivial<A> && Trivial<B>
        : first(A{first}), second(B{second}) {
    }

    explicit Pair(A&& first, B&& second)
        requires Move_Constructable<A> && Move_Constructable<B>
        : first(std::move(first)), second(std::move(second)) {
    }

    ~Pair() = default;

    Pair(const Pair& src)
        requires Trivial<A> && Trivial<B>
    = default;
    Pair& operator=(const Pair& src)
        requires Trivial<A> && Trivial<B>
    = default;

    Pair(Pair&& src) = default;
    Pair& operator=(Pair&& src) = default;

    Pair<A, B> clone() const
        requires Clone<A> && Clone<B>
    {
        return Pair<A, B>{first.clone(), second.clone()};
    }

    template<u64 Index>
    auto& get() {
        if constexpr(Index == 0) return first;
        if constexpr(Index == 1) return second;
    }
    template<u64 Index>
    const auto& get() const {
        if constexpr(Index == 0) return first;
        if constexpr(Index == 1) return second;
    }

    A first;
    B second;
};

template<typename A, typename B>
struct Reflect<Pair<A, B>> {
    using T = Pair<A, B>;
    static constexpr Literal name = "Pair";
    static constexpr Kind kind = Kind::record_;
    using members = List<FIELD(first), FIELD(second)>;
    static_assert(Record<T>);
};

} // namespace rpp
