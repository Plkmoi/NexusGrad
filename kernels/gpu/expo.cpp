// const char* op_name(Op o) {
//   static constexpr std::array<const char*, OpCount> names = {{
//   #define FLASH(name, arity, str) str,
//   #include "ad/detail/ops.def"
//   #undef OP
//   }};
//   return names[static_cast<std::size_t>(o)];
// }