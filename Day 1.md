
# C++ 템플릿 / Format 핵심 정리


```cpp

// 1. 가변 템플릿 출력 함수
// 역할: 여러 타입의 인자를 받아 문자열로 포맷 후 출력

template<typename... Args>
void print(string_view fmt_str, Args&&... args)
{
    auto fmt_args = make_format_args(args...); // 여러 인자를 포맷용으로 묶음
    auto result = vformat(fmt_str, fmt_args);   // 문자열 생성
    fputs(result.c_str(), stdout);              // 실제 출력
}
```

```cpp
// 2. 사용자 정의 타입 Formatter
// 역할: Frac 구조체를 "분자/분모" 형태로 출력


template<>
struct formatter<Frac>
{
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin(); // 별도 포맷 옵션 없음
    }

    template<typename FormatContext>
    auto format(const Frac& f, FormatContext& ctx)
    {
        return format_to(ctx.out(), "{}/{}", f.n, f.d);
    }
};
```
```cpp

int x{-3};                                // 중괄호 초기화

using UT = make_unsigned_t<T>;            // T의 unsigned 타입

std::cout;                                // std = namespace
using namespace std;                      // std:: 생략
```
