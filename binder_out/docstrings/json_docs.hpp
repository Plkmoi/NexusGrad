/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_nlohmann_json_abi_v3_12_0_adl_serializer =
R"doc(! default JSONSerializer template argument

This serializer ignores the template arguments and uses ADL
([argument-dependent
lookup](https://en.cppreference.com/w/cpp/language/adl)) for
serialization.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_adl_serializer_2 =
R"doc(! default JSONSerializer template argument

This serializer ignores the template arguments and uses ADL
([argument-dependent
lookup](https://en.cppreference.com/w/cpp/language/adl)) for
serialization.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_adl_serializer_from_json =
R"doc(convert a JSON value to any value type

See also:
    https://json.nlohmann.me/api/adl_serializer/from_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_adl_serializer_from_json_2 =
R"doc(convert a JSON value to any value type

See also:
    https://json.nlohmann.me/api/adl_serializer/from_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_adl_serializer_to_json =
R"doc(convert any value type to a JSON value

See also:
    https://json.nlohmann.me/api/adl_serializer/to_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json =
R"doc(a class to store JSON values

See also:
    https://json.nlohmann.me/api/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_2 =
R"doc(a class to store JSON values

See also:
    https://json.nlohmann.me/api/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_accept =
R"doc(check if the input is valid JSON

See also:
    https://json.nlohmann.me/api/basic_json/accept/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_accept_2 =
R"doc(check if the input is valid JSON

See also:
    https://json.nlohmann.me/api/basic_json/accept/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_accept_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_array =
R"doc(explicitly create an array from an initializer list

See also:
    https://json.nlohmann.me/api/basic_json/array/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_assert_invariant =
R"doc(! checks the class invariants

This function asserts the class invariants. It needs to be called at
the end of every constructor to make sure that created objects respect
the invariant. Furthermore, it has to be called each time the type of
a JSON value is changed, because the invariant expresses a
relationship between *m_type* and *m_value*.

Furthermore, the parent relation is checked for arrays and objects: If
*check_parents* true and the value is an array or object, then the
container's elements must have the current value as parent.

Parameter ``check_parents``:
    whether the parent relation should be checked. The value is true
    by default and should only be set to false during destruction of
    objects when the invariant does not need to hold.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at =
R"doc(access specified array element with bounds checking

See also:
    https://json.nlohmann.me/api/basic_json/at/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_2 =
R"doc(access specified array element with bounds checking

See also:
    https://json.nlohmann.me/api/basic_json/at/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_3 =
R"doc(access specified object element with bounds checking

See also:
    https://json.nlohmann.me/api/basic_json/at/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_4 =
R"doc(access specified object element with bounds checking

See also:
    https://json.nlohmann.me/api/basic_json/at/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_5 =
R"doc(access specified object element with bounds checking

See also:
    https://json.nlohmann.me/api/basic_json/at/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_6 =
R"doc(access specified object element with bounds checking

See also:
    https://json.nlohmann.me/api/basic_json/at/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_7 =
R"doc(access specified element via JSON Pointer

See also:
    https://json.nlohmann.me/api/basic_json/at/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_8 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_9 =
R"doc(access specified element via JSON Pointer

See also:
    https://json.nlohmann.me/api/basic_json/at/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_at_10 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_back =
R"doc(access the last element

See also:
    https://json.nlohmann.me/api/basic_json/back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_back_2 =
R"doc(access the last element

See also:
    https://json.nlohmann.me/api/basic_json/back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json =
R"doc(create an empty value with a given type

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_2 =
R"doc(create a null object

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_3 =
R"doc(create a JSON value from compatible types

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_4 =
R"doc(create a JSON value from an existing one

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_5 =
R"doc(create a container (array or object) from an initializer list

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_6 =
R"doc(construct an array with count copies of given value

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_7 =
R"doc(construct a JSON container given an iterator range

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_8 = R"doc(////////////////////////////////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_9 =
R"doc(copy constructor

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_basic_json_10 =
R"doc(move constructor

See also:
    https://json.nlohmann.me/api/basic_json/basic_json/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_begin =
R"doc(returns an iterator to the first element

See also:
    https://json.nlohmann.me/api/basic_json/begin/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_begin_2 =
R"doc(returns an iterator to the first element

See also:
    https://json.nlohmann.me/api/basic_json/begin/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_binary =
R"doc(explicitly create a binary array (without subtype)

See also:
    https://json.nlohmann.me/api/basic_json/binary/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_binary_2 =
R"doc(explicitly create a binary array (with subtype)

See also:
    https://json.nlohmann.me/api/basic_json/binary/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_binary_3 =
R"doc(explicitly create a binary array

See also:
    https://json.nlohmann.me/api/basic_json/binary/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_binary_4 =
R"doc(explicitly create a binary array (with subtype)

See also:
    https://json.nlohmann.me/api/basic_json/binary/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_cbegin =
R"doc(returns a const iterator to the first element

See also:
    https://json.nlohmann.me/api/basic_json/cbegin/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_cend =
R"doc(returns an iterator to one past the last element

See also:
    https://json.nlohmann.me/api/basic_json/cend/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_clear =
R"doc(clears the contents

See also:
    https://json.nlohmann.me/api/basic_json/clear/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_compares_unordered = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_compares_unordered_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_contains =
R"doc(check the existence of an element in a JSON object

See also:
    https://json.nlohmann.me/api/basic_json/contains/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_contains_2 =
R"doc(check the existence of an element in a JSON object

See also:
    https://json.nlohmann.me/api/basic_json/contains/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_contains_3 =
R"doc(check the existence of an element in a JSON object given a JSON
pointer

See also:
    https://json.nlohmann.me/api/basic_json/contains/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_contains_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_count =
R"doc(returns the number of occurrences of a key in a JSON object

See also:
    https://json.nlohmann.me/api/basic_json/count/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_count_2 =
R"doc(returns the number of occurrences of a key in a JSON object

See also:
    https://json.nlohmann.me/api/basic_json/count/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_crbegin =
R"doc(returns a const reverse iterator to the last element

See also:
    https://json.nlohmann.me/api/basic_json/crbegin/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_create = R"doc(helper for exception-safe object creation)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_crend =
R"doc(returns a const reverse iterator to one before the first

See also:
    https://json.nlohmann.me/api/basic_json/crend/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data = R"doc(///////////////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_data = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_data_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_data_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_data_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_data_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_m_type = R"doc(the type of the current element)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_m_value = R"doc(the value of the current element)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_data_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_diff =
R"doc(creates a diff as a JSON patch

See also:
    https://json.nlohmann.me/api/basic_json/diff/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_dump =
R"doc(serialization

See also:
    https://json.nlohmann.me/api/basic_json/dump/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_emplace =
R"doc(add an object to an object if key does not exist

See also:
    https://json.nlohmann.me/api/basic_json/emplace/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_emplace_back =
R"doc(add an object to an array

See also:
    https://json.nlohmann.me/api/basic_json/emplace_back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_empty =
R"doc(checks whether the container is empty.

See also:
    https://json.nlohmann.me/api/basic_json/empty/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_end =
R"doc(returns an iterator to one past the last element

See also:
    https://json.nlohmann.me/api/basic_json/end/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_end_2 =
R"doc(returns an iterator to one past the last element

See also:
    https://json.nlohmann.me/api/basic_json/end/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_erase =
R"doc(remove element given an iterator

See also:
    https://json.nlohmann.me/api/basic_json/erase/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_erase_2 =
R"doc(remove elements given an iterator range

See also:
    https://json.nlohmann.me/api/basic_json/erase/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_erase_3 =
R"doc(remove element from a JSON object given a key

See also:
    https://json.nlohmann.me/api/basic_json/erase/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_erase_4 =
R"doc(remove element from a JSON object given a key

See also:
    https://json.nlohmann.me/api/basic_json/erase/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_erase_5 =
R"doc(remove element from a JSON array given an index

See also:
    https://json.nlohmann.me/api/basic_json/erase/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_erase_internal = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_erase_internal_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_find =
R"doc(find an element in a JSON object

See also:
    https://json.nlohmann.me/api/basic_json/find/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_find_2 =
R"doc(find an element in a JSON object

See also:
    https://json.nlohmann.me/api/basic_json/find/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_find_3 =
R"doc(find an element in a JSON object

See also:
    https://json.nlohmann.me/api/basic_json/find/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_find_4 =
R"doc(find an element in a JSON object

See also:
    https://json.nlohmann.me/api/basic_json/find/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_flatten =
R"doc(return flattened JSON value

See also:
    https://json.nlohmann.me/api/basic_json/flatten/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_bjdata =
R"doc(create a JSON value from an input in BJData format

See also:
    https://json.nlohmann.me/api/basic_json/from_bjdata/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_bjdata_2 =
R"doc(create a JSON value from an input in BJData format

See also:
    https://json.nlohmann.me/api/basic_json/from_bjdata/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_bson =
R"doc(create a JSON value from an input in BSON format

See also:
    https://json.nlohmann.me/api/basic_json/from_bson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_bson_2 =
R"doc(create a JSON value from an input in BSON format

See also:
    https://json.nlohmann.me/api/basic_json/from_bson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_bson_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_bson_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_cbor =
R"doc(create a JSON value from an input in CBOR format

See also:
    https://json.nlohmann.me/api/basic_json/from_cbor/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_cbor_2 =
R"doc(create a JSON value from an input in CBOR format

See also:
    https://json.nlohmann.me/api/basic_json/from_cbor/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_cbor_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_cbor_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_msgpack =
R"doc(create a JSON value from an input in MessagePack format

See also:
    https://json.nlohmann.me/api/basic_json/from_msgpack/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_msgpack_2 =
R"doc(create a JSON value from an input in MessagePack format

See also:
    https://json.nlohmann.me/api/basic_json/from_msgpack/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_msgpack_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_msgpack_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_ubjson =
R"doc(create a JSON value from an input in UBJSON format

See also:
    https://json.nlohmann.me/api/basic_json/from_ubjson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_ubjson_2 =
R"doc(create a JSON value from an input in UBJSON format

See also:
    https://json.nlohmann.me/api/basic_json/from_ubjson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_ubjson_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_from_ubjson_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_front =
R"doc(access the first element

See also:
    https://json.nlohmann.me/api/basic_json/front/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_front_2 =
R"doc(access the first element

See also:
    https://json.nlohmann.me/api/basic_json/front/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get =
R"doc(! get a (pointer) value (explicit)

Performs explicit type conversion between the JSON value and a
compatible value if required.

- If the requested type is a pointer to the internally stored JSON
value that pointer is returned. No copies are made.

- If the requested type is the current basic_json, or a different
basic_json convertible from the current basic_json.

- Otherwise the value is converted by calling the
json_serializer<ValueType> `from_json()` method.

Template parameter ``ValueTypeCV``:
    the provided value type

Template parameter ``ValueType``:
    the returned value type

Returns:
    copy of the JSON value, converted to

Template parameter ``ValueType``:
    if necessary

Throws:
    what json_serializer<ValueType> `from_json()` method throws if
    conversion is required

.. versionadded:: version 2.1.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_2 =
R"doc(! get a pointer value (explicit)

Explicit pointer access to the internally stored JSON value. No copies
are made.

.. warning::
    The pointer becomes invalid if the underlying JSON object changes.

Template parameter ``PointerType``:
    pointer type; must be a pointer to array_t, object_t, string_t,
    boolean_t, number_integer_t, number_unsigned_t, or number_float_t.

Returns:
    pointer to the internally stored JSON value if the requested
    pointer type *PointerType* fits to the JSON value; `nullptr`
    otherwise

@complexity Constant.

@liveexample{The example below shows how pointers to internal values
of a JSON value can be requested. Note that no type conversions are
made and a `nullptr` is returned if the value and the requested
pointer type does not match.,get__PointerType}

See also:
    see get_ptr() for explicit pointer-member access

.. versionadded:: version 1.0.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_allocator =
R"doc(returns the allocator associated with the container

See also:
    https://json.nlohmann.me/api/basic_json/get_allocator/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_binary =
R"doc(get a binary value

See also:
    https://json.nlohmann.me/api/basic_json/get_binary/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_binary_2 =
R"doc(get a binary value

See also:
    https://json.nlohmann.me/api/basic_json/get_binary/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl = R"doc(get a boolean (explicit))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_2 =
R"doc(! get a value (explicit)

Explicit type conversion between the JSON value and a compatible value
which is [CopyConstructible](https://en.cppreference.com/w/cpp/named_r
eq/CopyConstructible) and [DefaultConstructible](https://en.cppreferen
ce.com/w/cpp/named_req/DefaultConstructible). The value is converted
by calling the json_serializer<ValueType> `from_json()` method.

The function is equivalent to executing

```
{.cpp}
ValueType ret;
JSONSerializer<ValueType>::from_json(*this, ret);
return ret;
```

This overloads is chosen if: - *ValueType* is not basic_json, -
json_serializer<ValueType> has a `from_json()` method of the form
`void from_json(const basic_json&, ValueType&)`, and -
json_serializer<ValueType> does not have a `from_json()` method of the
form `ValueType from_json(const basic_json&)`

Template parameter ``ValueType``:
    the returned value type

Returns:
    copy of the JSON value, converted to *ValueType*

Throws:
    what json_serializer<ValueType> `from_json()` method throws

@liveexample{The example below shows several conversions from JSON
values to other types. There a few things to note: (1) Floating-point
numbers can be converted to integers\, (2) A JSON array can be
converted to a standard `std::vector<short>`\, (3) A JSON object can
be converted to C++ associative containers such as
`std::unordered_map<std::string\, json>`.,get__ValueType_const}

.. versionadded:: version 2.1.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_3 =
R"doc(! get a value (explicit); special case

Explicit type conversion between the JSON value and a compatible value
which is **not** [CopyConstructible](https://en.cppreference.com/w/cpp
/named_req/CopyConstructible) and **not** [DefaultConstructible](https
://en.cppreference.com/w/cpp/named_req/DefaultConstructible). The
value is converted by calling the json_serializer<ValueType>
`from_json()` method.

The function is equivalent to executing

```
{.cpp}
return JSONSerializer<ValueType>::from_json(*this);
```

This overloads is chosen if: - *ValueType* is not basic_json and -
json_serializer<ValueType> has a `from_json()` method of the form
`ValueType from_json(const basic_json&)`

@note If json_serializer<ValueType> has both overloads of
`from_json()`, this one is chosen.

Template parameter ``ValueType``:
    the returned value type

Returns:
    copy of the JSON value, converted to *ValueType*

Throws:
    what json_serializer<ValueType> `from_json()` method throws

.. versionadded:: version 2.1.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_4 =
R"doc(! get special-case overload

This overloads converts the current basic_json in a different
basic_json type

Template parameter ``BasicJsonType``:
    == basic_json

Returns:
    a copy of *this, converted into *BasicJsonType*

@complexity Depending on the implementation of the called
`from_json()` method.

.. versionadded:: version 3.2.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_5 =
R"doc(! get special-case overload

This overloads avoids a lot of template boilerplate, it can be seen as
the identity method

Template parameter ``BasicJsonType``:
    == basic_json

Returns:
    a copy of *this

@complexity Constant.

.. versionadded:: version 2.1.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_6 = R"doc(! get a pointer value (explicit) @copydoc get())doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr = R"doc(get a pointer to the value (object))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_2 = R"doc(get a pointer to the value (object))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_3 = R"doc(get a pointer to the value (array))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_4 = R"doc(get a pointer to the value (array))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_5 = R"doc(get a pointer to the value (string))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_6 = R"doc(get a pointer to the value (string))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_7 = R"doc(get a pointer to the value (boolean))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_8 = R"doc(get a pointer to the value (boolean))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_9 = R"doc(get a pointer to the value (integer number))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_10 = R"doc(get a pointer to the value (integer number))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_11 = R"doc(get a pointer to the value (unsigned number))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_12 = R"doc(get a pointer to the value (unsigned number))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_13 = R"doc(get a pointer to the value (floating-point number))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_14 = R"doc(get a pointer to the value (floating-point number))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_15 = R"doc(get a pointer to the value (binary))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_impl_ptr_16 = R"doc(get a pointer to the value (binary))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_ptr =
R"doc(get a pointer value (implicit)

See also:
    https://json.nlohmann.me/api/basic_json/get_ptr/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_ptr_2 =
R"doc(get a pointer value (implicit)

See also:
    https://json.nlohmann.me/api/basic_json/get_ptr/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_ref =
R"doc(get a reference value (implicit)

See also:
    https://json.nlohmann.me/api/basic_json/get_ref/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_ref_2 =
R"doc(get a reference value (implicit)

See also:
    https://json.nlohmann.me/api/basic_json/get_ref/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_ref_impl =
R"doc(! helper function to implement get_ref()

This function helps to implement get_ref() without code duplication
for const and non-const overloads

Template parameter ``ThisType``:
    will be deduced as `basic_json` or `const basic_json`

Throws:
    type_error.303 if ReferenceType does not match underlying value
    type of the current JSON)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_to =
R"doc(get a value (explicit)

See also:
    https://json.nlohmann.me/api/basic_json/get_to/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_to_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_get_to_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_insert =
R"doc(inserts element into array

See also:
    https://json.nlohmann.me/api/basic_json/insert/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_insert_2 =
R"doc(inserts element into array

See also:
    https://json.nlohmann.me/api/basic_json/insert/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_insert_3 =
R"doc(inserts copies of element into array

See also:
    https://json.nlohmann.me/api/basic_json/insert/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_insert_4 =
R"doc(inserts range of elements into array

See also:
    https://json.nlohmann.me/api/basic_json/insert/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_insert_5 =
R"doc(inserts elements from initializer list into array

See also:
    https://json.nlohmann.me/api/basic_json/insert/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_insert_6 =
R"doc(inserts range of elements into object

See also:
    https://json.nlohmann.me/api/basic_json/insert/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_insert_iterator =
R"doc(Helper for insertion of an iterator @note: This uses std::distance to
support GCC 4.8, see https://github.com/nlohmann/json/pull/1257)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_array =
R"doc(return whether value is an array

See also:
    https://json.nlohmann.me/api/basic_json/is_array/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_binary =
R"doc(return whether value is a binary array

See also:
    https://json.nlohmann.me/api/basic_json/is_binary/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_boolean =
R"doc(return whether value is a boolean

See also:
    https://json.nlohmann.me/api/basic_json/is_boolean/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_discarded =
R"doc(return whether value is discarded

See also:
    https://json.nlohmann.me/api/basic_json/is_discarded/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_null =
R"doc(return whether value is null

See also:
    https://json.nlohmann.me/api/basic_json/is_null/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_number =
R"doc(return whether value is a number

See also:
    https://json.nlohmann.me/api/basic_json/is_number/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_number_float =
R"doc(return whether value is a floating-point number

See also:
    https://json.nlohmann.me/api/basic_json/is_number_float/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_number_integer =
R"doc(return whether value is an integer number

See also:
    https://json.nlohmann.me/api/basic_json/is_number_integer/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_number_unsigned =
R"doc(return whether value is an unsigned integer number

See also:
    https://json.nlohmann.me/api/basic_json/is_number_unsigned/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_object =
R"doc(return whether value is an object

See also:
    https://json.nlohmann.me/api/basic_json/is_object/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_primitive =
R"doc(return whether type is primitive

See also:
    https://json.nlohmann.me/api/basic_json/is_primitive/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_string =
R"doc(return whether value is a string

See also:
    https://json.nlohmann.me/api/basic_json/is_string/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_is_structured =
R"doc(return whether type is structured

See also:
    https://json.nlohmann.me/api/basic_json/is_structured/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_items =
R"doc(helper to access iterator member functions in range-based for

See also:
    https://json.nlohmann.me/api/basic_json/items/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_items_2 =
R"doc(helper to access iterator member functions in range-based for

See also:
    https://json.nlohmann.me/api/basic_json/items/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_iterator_wrapper =
R"doc(wrapper to access iterator member functions in range-based for

See also:
    https://json.nlohmann.me/api/basic_json/items/ @deprecated This
    function is deprecated since 3.1.0 and will be removed in version
    4.0.0 of the library. Please use items() instead; that is, replace
    `json::iterator_wrapper(j)` with `j.items()`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_iterator_wrapper_2 =
R"doc(wrapper to access iterator member functions in range-based for

See also:
    https://json.nlohmann.me/api/basic_json/items/ @deprecated This
    function is deprecated since 3.1.0 and will be removed in version
    4.0.0 of the library. Please use items() instead; that is, replace
    `json::iterator_wrapper(j)` with `j.items()`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_m_data = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_max_size =
R"doc(returns the maximum possible number of elements

See also:
    https://json.nlohmann.me/api/basic_json/max_size/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_merge_patch =
R"doc(applies a JSON Merge Patch

See also:
    https://json.nlohmann.me/api/basic_json/merge_patch/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_meta =
R"doc(returns version information on the library

See also:
    https://json.nlohmann.me/api/basic_json/meta/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_object =
R"doc(explicitly create an object from an initializer list

See also:
    https://json.nlohmann.me/api/basic_json/object/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array =
R"doc(access specified array element

See also:
    https://json.nlohmann.me/api/basic_json/operator%5B%5D/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_2 =
R"doc(access specified array element

See also:
    https://json.nlohmann.me/api/basic_json/operator%5B%5D/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_3 =
R"doc(access specified object element

See also:
    https://json.nlohmann.me/api/basic_json/operator%5B%5D/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_4 =
R"doc(access specified object element

See also:
    https://json.nlohmann.me/api/basic_json/operator%5B%5D/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_7 =
R"doc(access specified object element

See also:
    https://json.nlohmann.me/api/basic_json/operator%5B%5D/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_8 =
R"doc(access specified object element

See also:
    https://json.nlohmann.me/api/basic_json/operator%5B%5D/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_9 =
R"doc(access specified element via JSON Pointer

See also:
    https://json.nlohmann.me/api/basic_json/operator%5B%5D/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_10 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_11 =
R"doc(access specified element via JSON Pointer

See also:
    https://json.nlohmann.me/api/basic_json/operator%5B%5D/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_array_12 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_assign =
R"doc(copy assignment

See also:
    https://json.nlohmann.me/api/basic_json/operator=/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_iadd =
R"doc(add an object to an array

See also:
    https://json.nlohmann.me/api/basic_json/operator+=/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_iadd_2 =
R"doc(add an object to an array

See also:
    https://json.nlohmann.me/api/basic_json/operator+=/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_iadd_3 =
R"doc(add an object to an object

See also:
    https://json.nlohmann.me/api/basic_json/operator+=/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_iadd_4 =
R"doc(add an object to an object

See also:
    https://json.nlohmann.me/api/basic_json/operator+=/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_nlohmann_detail_value_t =
R"doc(return the type of the JSON value (implicit)

See also:
    https://json.nlohmann.me/api/basic_json/operator_value_t/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_operator_type_parameter_1_0 =
R"doc(! get a value (implicit)

Implicit type conversion between the JSON value and a compatible
value. The call is realized by calling get() const.

Template parameter ``ValueType``:
    non-pointer type compatible to the JSON value, for instance `int`
    for JSON integer numbers, `bool` for JSON booleans, or
    `std::vector` types for JSON arrays. The character type of
    string_t as well as an initializer list of this type is excluded
    to avoid ambiguities as these types implicitly convert to
    `std::string`.

Returns:
    copy of the JSON value, converted to type *ValueType*

Throws:
    type_error.302 in case passed type *ValueType* is incompatible to
    the JSON value type (e.g., the JSON value is of type boolean, but
    a string is requested); see example below

@complexity Linear in the size of the JSON value.

@liveexample{The example below shows several conversions from JSON
values to other types. There a few things to note: (1) Floating-point
numbers can be converted to integers\, (2) A JSON array can be
converted to a standard `std::vector<short>`\, (3) A JSON object can
be converted to C++ associative containers such as
`std::unordered_map<std::string\, json>`.,operator__ValueType}

.. versionadded:: version 1.0.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_parse =
R"doc(deserialize from a compatible input

See also:
    https://json.nlohmann.me/api/basic_json/parse/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_parse_2 =
R"doc(deserialize from a pair of character iterators

See also:
    https://json.nlohmann.me/api/basic_json/parse/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_parse_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_parser = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_patch =
R"doc(applies a JSON patch to a copy of the current object

See also:
    https://json.nlohmann.me/api/basic_json/patch/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_patch_inplace =
R"doc(applies a JSON patch in-place without copying the object

See also:
    https://json.nlohmann.me/api/basic_json/patch/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_push_back =
R"doc(add an object to an array

See also:
    https://json.nlohmann.me/api/basic_json/push_back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_push_back_2 =
R"doc(add an object to an array

See also:
    https://json.nlohmann.me/api/basic_json/push_back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_push_back_3 =
R"doc(add an object to an object

See also:
    https://json.nlohmann.me/api/basic_json/push_back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_push_back_4 =
R"doc(add an object to an object

See also:
    https://json.nlohmann.me/api/basic_json/push_back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_rbegin =
R"doc(returns an iterator to the reverse-beginning

See also:
    https://json.nlohmann.me/api/basic_json/rbegin/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_rbegin_2 =
R"doc(returns an iterator to the reverse-beginning

See also:
    https://json.nlohmann.me/api/basic_json/rbegin/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_rend =
R"doc(returns an iterator to the reverse-end

See also:
    https://json.nlohmann.me/api/basic_json/rend/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_rend_2 =
R"doc(returns an iterator to the reverse-end

See also:
    https://json.nlohmann.me/api/basic_json/rend/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_sax_parse =
R"doc(generate SAX events

See also:
    https://json.nlohmann.me/api/basic_json/sax_parse/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_sax_parse_2 =
R"doc(generate SAX events

See also:
    https://json.nlohmann.me/api/basic_json/sax_parse/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_sax_parse_3 =
R"doc(generate SAX events

See also:
    https://json.nlohmann.me/api/basic_json/sax_parse/ @deprecated
    This function is deprecated since 3.8.0 and will be removed in
    version 4.0.0 of the library. Please use sax_parse(ptr, ptr + len)
    instead.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_set_parent = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_set_parents = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_set_parents_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_size =
R"doc(returns the number of elements

See also:
    https://json.nlohmann.me/api/basic_json/size/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_swap =
R"doc(exchanges the values

See also:
    https://json.nlohmann.me/api/basic_json/swap/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_swap_2 =
R"doc(exchanges the values

See also:
    https://json.nlohmann.me/api/basic_json/swap/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_swap_3 =
R"doc(exchanges the values

See also:
    https://json.nlohmann.me/api/basic_json/swap/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_swap_4 =
R"doc(exchanges the values

See also:
    https://json.nlohmann.me/api/basic_json/swap/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_swap_5 =
R"doc(exchanges the values

See also:
    https://json.nlohmann.me/api/basic_json/swap/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_swap_6 =
R"doc(exchanges the values

See also:
    https://json.nlohmann.me/api/basic_json/swap/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_bjdata =
R"doc(create a BJData serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_bjdata/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_bjdata_2 =
R"doc(create a BJData serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_bjdata/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_bjdata_3 =
R"doc(create a BJData serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_bjdata/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_bson =
R"doc(create a BSON serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_bson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_bson_2 =
R"doc(create a BSON serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_bson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_bson_3 =
R"doc(create a BSON serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_bson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_cbor =
R"doc(create a CBOR serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_cbor/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_cbor_2 =
R"doc(create a CBOR serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_cbor/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_cbor_3 =
R"doc(create a CBOR serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_cbor/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_msgpack =
R"doc(create a MessagePack serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_msgpack/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_msgpack_2 =
R"doc(create a MessagePack serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_msgpack/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_msgpack_3 =
R"doc(create a MessagePack serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_msgpack/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_ubjson =
R"doc(create a UBJSON serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_ubjson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_ubjson_2 =
R"doc(create a UBJSON serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_ubjson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_to_ubjson_3 =
R"doc(create a UBJSON serialization of a given JSON value

See also:
    https://json.nlohmann.me/api/basic_json/to_ubjson/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_type =
R"doc(return the type of the JSON value (explicit)

See also:
    https://json.nlohmann.me/api/basic_json/type/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_type_name =
R"doc(return the type as string

See also:
    https://json.nlohmann.me/api/basic_json/type_name/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_unflatten =
R"doc(unflatten a previously flattened JSON value

See also:
    https://json.nlohmann.me/api/basic_json/unflatten/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_update =
R"doc(updates a JSON object from another object, overwriting existing keys

See also:
    https://json.nlohmann.me/api/basic_json/update/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_update_2 =
R"doc(updates a JSON object from another object, overwriting existing keys

See also:
    https://json.nlohmann.me/api/basic_json/update/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_value =
R"doc(access specified object element with default value

See also:
    https://json.nlohmann.me/api/basic_json/value/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_value_2 =
R"doc(access specified object element with default value

See also:
    https://json.nlohmann.me/api/basic_json/value/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_value_3 =
R"doc(access specified object element with default value

See also:
    https://json.nlohmann.me/api/basic_json/value/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_value_4 =
R"doc(access specified object element via JSON Pointer with default value

See also:
    https://json.nlohmann.me/api/basic_json/value/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_value_5 =
R"doc(access specified object element via JSON Pointer with default value

See also:
    https://json.nlohmann.me/api/basic_json/value/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_value_6 =
R"doc(access specified object element via JSON Pointer with default value

See also:
    https://json.nlohmann.me/api/basic_json/value/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_value_7 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_basic_json_value_8 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype =
R"doc(an internal type for a backed binary type

See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_byte_container_with_subtype =
R"doc(See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/byte_cont
    ainer_with_subtype/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_byte_container_with_subtype_2 =
R"doc(See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/byte_cont
    ainer_with_subtype/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_byte_container_with_subtype_3 =
R"doc(See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/byte_cont
    ainer_with_subtype/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_byte_container_with_subtype_4 =
R"doc(See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/byte_cont
    ainer_with_subtype/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_byte_container_with_subtype_5 =
R"doc(See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/byte_cont
    ainer_with_subtype/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_clear_subtype =
R"doc(clears the binary subtype

See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/clear_sub
    type/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_has_subtype =
R"doc(return whether the value has a subtype

See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/has_subty
    pe/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_m_has_subtype = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_m_subtype = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_operator_eq = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_operator_ne = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_set_subtype =
R"doc(sets the binary subtype

See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/set_subty
    pe/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_byte_container_with_subtype_subtype =
R"doc(return the binary subtype

See also:
    https://json.nlohmann.me/api/byte_container_with_subtype/subtype/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail2_begin = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail2_begin_tag = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail2_end = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail2_end_tag = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail2_would_call_std_begin = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail2_would_call_std_end = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_actual_object_comparator = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader = R"doc(! deserialization of CBOR, MessagePack, and UBJSON values)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_binary_reader =
R"doc(! create a binary reader

Parameter ``adapter``:
    input adapter to read from)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_binary_reader_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_binary_reader_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_bjd_optimized_type_markers = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_bjd_types_map = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_byte_swap = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_chars_read = R"doc(the number of characters read)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_current = R"doc(the current character)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_exception_message =
R"doc(!

Parameter ``format``:
    the current format

Parameter ``detail``:
    a detailed error message

Parameter ``context``:
    further context information

Returns:
    a message string to use in the parse_error exceptions)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get =
R"doc(! get next character from the input

This function provides the interface to the used input adapter. It
does not throw in case the input reached EOF, but returns a -'ve
valued `char_traits<char_type>::eof()` in that case.

Returns:
    character read from the input)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_binary =
R"doc(! create a byte array by reading bytes from the input

Template parameter ``NumberType``:
    the type of the number

Parameter ``format``:
    the current format (for diagnostics)

Parameter ``len``:
    number of bytes to read

Parameter ``result``:
    byte array created by reading *len* bytes

Returns:
    whether byte array creation completed

@note We can not reserve *len* bytes for the result, because *len* may
be too large. Usually, unexpect_eof() detects the end of the input
before we run out of memory.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_bson_binary =
R"doc(! Parses a byte array input of length *len* from the BSON input.

Parameter ``len``:
    The length of the byte array to be read.

Parameter ``result``:
    A reference to the binary variable where the read array is to be
    stored.

Template parameter ``NumberType``:
    The type of the length *len* @pre len >= 0

Returns:
    `true` if the byte array was successfully parsed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_bson_cstr =
R"doc(! Parses a C-style string from the BSON input.

Parameter ``result``:
    A reference to the string variable where the read string is to be
    stored.

Returns:
    `true` if the \x00-byte indicating the end of the string was
    encountered before the EOF; false` indicates an unexpected EOF.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_bson_string =
R"doc(! Parses a zero-terminated string of length *len* from the BSON input.

Parameter ``len``:
    The length (including the zero-byte at the end) of the string to
    be read.

Parameter ``result``:
    A reference to the string variable where the read string is to be
    stored.

Template parameter ``NumberType``:
    The type of the length *len* @pre len >= 1

Returns:
    `true` if the string was successfully parsed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_cbor_array =
R"doc(!

Parameter ``len``:
    the length of the array or detail::unknown_size() for an array of
    indefinite size

Parameter ``tag_handler``:
    how CBOR tags should be treated

Returns:
    whether array creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_cbor_binary =
R"doc(! reads a CBOR byte array

This function first reads starting bytes to determine the expected
byte array length and then copies this number of bytes into the byte
array. Additionally, CBOR's byte arrays with indefinite lengths are
supported.

Parameter ``result``:
    created byte array

Returns:
    whether byte array creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_cbor_object =
R"doc(!

Parameter ``len``:
    the length of the object or detail::unknown_size() for an object
    of indefinite size

Parameter ``tag_handler``:
    how CBOR tags should be treated

Returns:
    whether object creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_cbor_string =
R"doc(! reads a CBOR string

This function first reads starting bytes to determine the expected
string length and then copies this number of bytes into a string.
Additionally, CBOR's strings with indefinite lengths are supported.

Parameter ``result``:
    created string

Returns:
    whether string creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ignore_noop =
R"doc(!

Returns:
    character read from the input after ignoring all 'N' entries)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_msgpack_array =
R"doc(!

Parameter ``len``:
    the length of the array

Returns:
    whether array creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_msgpack_binary =
R"doc(! reads a MessagePack byte array

This function first reads starting bytes to determine the expected
byte array length and then copies this number of bytes into a byte
array.

Parameter ``result``:
    created byte array

Returns:
    whether byte array creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_msgpack_object =
R"doc(!

Parameter ``len``:
    the length of the object

Returns:
    whether object creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_msgpack_string =
R"doc(! reads a MessagePack string

This function first reads starting bytes to determine the expected
string length and then copies this number of bytes into a string.

Parameter ``result``:
    created string

Returns:
    whether string creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_number = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_string =
R"doc(! create a string by reading characters from the input

Template parameter ``NumberType``:
    the type of the number

Parameter ``format``:
    the current format (for diagnostics)

Parameter ``len``:
    number of characters to read

Parameter ``result``:
    string created by reading *len* bytes

Returns:
    whether string creation completed

@note We can not reserve *len* bytes for the result, because *len* may
be too large. Usually, unexpect_eof() detects the end of the input
before we run out of string memory.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_to =
R"doc(! get_to read into a primitive type

This function provides the interface to the used input adapter. It
does not throw in case the input reached EOF, but returns false
instead

Returns:
    bool, whether the read was successful)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_token_string =
R"doc(!

Returns:
    a string representation of the last read byte)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ubjson_array =
R"doc(!

Returns:
    whether array creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ubjson_high_precision_number = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ubjson_ndarray_size =
R"doc(!

Parameter ``dim``:
    an integer vector storing the ND array dimensions

Returns:
    whether reading ND array size vector is successful)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ubjson_object =
R"doc(!

Returns:
    whether object creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ubjson_size_type =
R"doc(! determine the type and size for a container

In the optimized UBJSON format, a type and a size can be provided to
allow for a more compact representation.

Parameter ``result``:
    pair of the size and the type

Parameter ``inside_ndarray``:
    whether the parser is parsing an ND array dimensional vector

Returns:
    whether pair creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ubjson_size_value =
R"doc(!

Parameter ``result``:
    determined size

Parameter ``is_ndarray``:
    for input, `true` means already inside an ndarray vector or
    ndarray dimension is not allowed; `false` means ndarray is
    allowed; for output, `true` means an ndarray is found; is_ndarray
    can only return `true` when its initial value is `false`

Parameter ``prefix``:
    type marker if already read, otherwise set to 0

Returns:
    whether size determination completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ubjson_string =
R"doc(! reads a UBJSON string

This function is either called after reading the 'S' byte explicitly
indicating a string, or in case of an object key where the 'S' byte
can be left out.

Parameter ``result``:
    created string

Parameter ``get_char``:
    whether a new character should be retrieved from the input (true,
    default) or whether the last read character should be considered
    instead

Returns:
    whether string creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_get_ubjson_value =
R"doc(!

Parameter ``prefix``:
    the previously read or set type prefix

Returns:
    whether value creation completed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_ia = R"doc(input adapter)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_input_format = R"doc(input format)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_is_little_endian = R"doc(whether we can assume little endianness)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_parse_bson_array =
R"doc(! Reads an array from the BSON input and passes it to the SAX-parser.

Returns:
    whether a valid BSON-array was passed to the SAX parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_parse_bson_element_internal =
R"doc(! Read a BSON document element of the given *element_type*.

Parameter ``element_type``:
    The BSON element type, c.f. http://bsonspec.org/spec.html

Parameter ``element_type_parse_position``:
    The position in the input stream, where the `element_type` was
    read. $.. warning::

Not all BSON element types are supported yet. An unsupported
*element_type* will give rise to a parse_error.114: Unsupported BSON
record type 0x...

Returns:
    whether a valid BSON-object/array was passed to the SAX parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_parse_bson_element_list =
R"doc(! Read a BSON element list (as specified in the BSON-spec)

The same binary layout is used for objects and arrays, hence it must
be indicated with the argument *is_array* which one is expected (true
--> array, false --> object).

Parameter ``is_array``:
    Determines if the element list being read is to be treated as an
    object (*is_array* == false), or as an array (*is_array* == true).

Returns:
    whether a valid BSON-object/array was passed to the SAX parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_parse_bson_internal =
R"doc(! Reads in a BSON-object and passes it to the SAX-parser.

Returns:
    whether a valid BSON-value was passed to the SAX parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_parse_cbor_internal =
R"doc(!

Parameter ``get_char``:
    whether a new character should be retrieved from the input (true)
    or whether the last read character should be considered instead
    (false)

Parameter ``tag_handler``:
    how CBOR tags should be treated

Returns:
    whether a valid CBOR value was passed to the SAX parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_parse_msgpack_internal =
R"doc(!

Returns:
    whether a valid MessagePack value was passed to the SAX parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_parse_ubjson_internal =
R"doc(!

Parameter ``get_char``:
    whether a new character should be retrieved from the input (true,
    default) or whether the last read character should be considered
    instead

Returns:
    whether a valid UBJSON value was passed to the SAX parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_sax = R"doc(the SAX parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_sax_parse =
R"doc(!

Parameter ``format``:
    the binary format to parse

Parameter ``sax_``:
    a SAX event processor

Parameter ``strict``:
    whether to expect the input to be consumed completed

Parameter ``tag_handler``:
    how to treat CBOR tags

Returns:
    whether parsing was successful)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_reader_unexpect_eof =
R"doc(!

Parameter ``format``:
    the current format (for diagnostics)

Parameter ``context``:
    further context information (for diagnostics)

Returns:
    whether the last read character is not EOF)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer = R"doc(! serialization to CBOR and MessagePack values)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_binary_writer =
R"doc(! create a binary writer

Parameter ``adapter``:
    output adapter to write to)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_calc_bson_array_size =
R"doc(!

Returns:
    The size of the BSON-encoded array *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_calc_bson_binary_size =
R"doc(!

Returns:
    The size of the BSON-encoded binary array *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_calc_bson_element_size =
R"doc(! Calculates the size necessary to serialize the JSON value *j* with
its *name*

Returns:
    The calculated size for the BSON document entry for *j* with the
    given *name*.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_calc_bson_entry_header_size =
R"doc(!

Returns:
    The size of a BSON document entry header, including the id marker
    and the entry name size (and its null-terminator).)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_calc_bson_integer_size =
R"doc(!

Returns:
    The size of the BSON-encoded integer *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_calc_bson_object_size =
R"doc(! Calculates the size of the BSON serialization of the given JSON-
object *j*.

Parameter ``value``:
    JSON value to serialize @pre value.type() == value_t::object)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_calc_bson_string_size =
R"doc(!

Returns:
    The size of the BSON-encoded string in *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_calc_bson_unsigned_size =
R"doc(!

Returns:
    The size of the BSON-encoded unsigned integer in *j*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_get_cbor_float_prefix = R"doc(///////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_get_cbor_float_prefix_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_get_msgpack_float_prefix = R"doc(//////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_get_msgpack_float_prefix_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_get_ubjson_float_prefix = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_get_ubjson_float_prefix_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_is_little_endian = R"doc(whether we can assume little endianness)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_oa = R"doc(the output)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_to_char_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_to_char_type_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_to_char_type_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_to_char_type_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_ubjson_prefix = R"doc(! determine the type prefix of container values)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bjdata_ndarray =
R"doc(!

Returns:
    false if the object is successfully converted to a bjdata ndarray,
    true if the type or size is invalid)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson =
R"doc(!

Parameter ``j``:
    JSON value to serialize @pre j.type() == value_t::object)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_array = R"doc(! Writes a BSON element with key *name* and array *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_binary = R"doc(! Writes a BSON element with key *name* and binary value *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_boolean = R"doc(! Writes a BSON element with key *name* and boolean value *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_double = R"doc(! Writes a BSON element with key *name* and double value *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_element =
R"doc(! Serializes the JSON value *j* to BSON and associates it with the key
*name*.

Parameter ``name``:
    The name to associate with the JSON entity *j* within the current
    BSON document)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_entry_header = R"doc(! Writes the given *element_type* and *name* to the output adapter)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_integer = R"doc(! Writes a BSON element with key *name* and integer *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_null = R"doc(! Writes a BSON element with key *name* and null value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_object =
R"doc(!

Parameter ``value``:
    JSON value to serialize @pre value.type() == value_t::object)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_object_entry = R"doc(! Writes a BSON element with key *name* and object *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_string = R"doc(! Writes a BSON element with key *name* and string value *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_bson_unsigned = R"doc(! Writes a BSON element with key *name* and unsigned *value*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_cbor =
R"doc(!

Parameter ``j``:
    JSON value to serialize)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_compact_float = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_msgpack =
R"doc(!

Parameter ``j``:
    JSON value to serialize)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_number = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_number_with_ubjson_prefix = R"doc(/////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_number_with_ubjson_prefix_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_number_with_ubjson_prefix_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_binary_writer_write_ubjson =
R"doc(!

Parameter ``j``:
    JSON value to serialize

Parameter ``use_count``:
    whether to use '#' prefixes (optimized format)

Parameter ``use_type``:
    whether to use '$' prefixes (optimized format)

Parameter ``add_prefix``:
    whether prefixes need to be used for this value

Parameter ``use_bjdata``:
    whether write in BJData format, default is false

Parameter ``bjdata_version``:
    which BJData version to use, default is draft2)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_bjdata_version_t = R"doc(how to encode BJData)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_bjdata_version_t_draft2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_bjdata_version_t_draft3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_cbor_tag_handler_t = R"doc(how to treat CBOR tags)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_cbor_tag_handler_t_error = R"doc(< throw a parse_error exception in case of a tag)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_cbor_tag_handler_t_ignore = R"doc(< ignore tags)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_cbor_tag_handler_t_store = R"doc(< store tags as binary type)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits = R"doc(//////////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits_eof = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits_eof_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits_to_char_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits_to_char_type_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits_to_int_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_char_traits_to_int_type_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_combine = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_into = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_into_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_into_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_into_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_into_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_into_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_into_7 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_into_8 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_length = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_length_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_length_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_length_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_length_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_concat_length_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_conditional_static_cast = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_conditional_static_cast_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_conjunction = R"doc(////////////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_container_input_adapter_factory_impl_container_input_adapter_factory = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_detector = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_append_exponent =
R"doc(! appends a decimal representation of e to buf

Returns:
    a pointer to the element following the exponent. @pre -1000 < e <
    1000)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_boundaries = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_boundaries_minus = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_boundaries_plus = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_boundaries_w = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_cached_power = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_cached_power_e = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_cached_power_f = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_cached_power_k = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_compute_boundaries =
R"doc(! Compute the (normalized) diyfp representing the input number 'value'
and its boundaries.

@pre value must be finite and positive)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_diyfp = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_diyfp_diyfp = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_diyfp_e = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_diyfp_f = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_diyfp_mul =
R"doc(! returns x * y @note The result is rounded. (Only the upper q bits
are returned.))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_diyfp_normalize = R"doc(! normalize x such that the significand is >= 2^(q-1) @pre x.f != 0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_diyfp_normalize_to =
R"doc(! normalize x such that the result has the exponent E @pre e >= x.e
and the upper e - x.e bits of x.f must be zero.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_diyfp_sub = R"doc(! returns x - y @pre x.e == y.e and x.f >= y.f)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_find_largest_pow10 =
R"doc(! For n != 0, returns k, such that pow10 := 10^(k-1) <= n < 10^k. For
n == 0, returns 1 and sets pow10 := 1.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_format_buffer =
R"doc(! prettify v = buf * 10^decimal_exponent

If v is in the range [10^min_exp, 10^max_exp) it will be printed in
fixed-point notation. Otherwise it will be printed in exponential
notation.

@pre min_exp < 0 @pre max_exp > 0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_get_cached_power_for_binary_exponent =
R"doc(! For a normalized diyfp w = f * 2^e, this function returns a
(normalized) cached power-of-ten c = f_c * 2^e_c, such that the
exponent of the product w * c satisfies (Definition 3.2 from [1])

alpha <= e_c + e + q <= gamma.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_grisu2 =
R"doc(! v = buf * 10^decimal_exponent len is the length of the buffer
(number of decimal digits) The buffer must be large enough, i.e. >=
max_digits10.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_grisu2_2 =
R"doc(! v = buf * 10^decimal_exponent len is the length of the buffer
(number of decimal digits) The buffer must be large enough, i.e. >=
max_digits10.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_grisu2_digit_gen =
R"doc(! Generates V = buffer * 10^decimal_exponent, such that M- <= V <= M+.
M- and M+ must be normalized and share the same exponent -60 <= e <=
-32.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_grisu2_round = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_dtoa_impl_reinterpret_bits = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_error_handler_t = R"doc(how to treat decoding errors)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_error_handler_t_ignore = R"doc(< ignore invalid UTF-8 sequences)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_error_handler_t_replace = R"doc(< replace invalid UTF-8 sequences with U+FFFD)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_error_handler_t_strict = R"doc(< throw a type_error exception in case of invalid UTF-8)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_escape =
R"doc(! string escaping as described in RFC 6901 (Sect. 4)

Parameter ``s``:
    string to escape

Returns:
    escaped string

Note the order of escaping "~" to "~0" and "/" to "~1" is important.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception =
R"doc(general exception of the basic_json class

See also:
    https://json.nlohmann.me/api/basic_json/exception/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception_diagnostics = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception_diagnostics_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception_exception = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception_get_byte_positions = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception_id = R"doc(the id of the exception)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception_m = R"doc(an exception object as storage for error messages)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception_name = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_exception_what = R"doc(returns the explanatory string)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor = R"doc(///////////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_7 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_8 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_9 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_7 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_8 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_9 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_10 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_11 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_12 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_13 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_14 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_15 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_16 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_external_constructor_construct_17 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter =
R"doc(! Input adapter for stdio file access. This adapter read only 1 byte
and do not use any buffer. This adapter is a very low level adapter.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter_file_input_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter_file_input_adapter_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter_file_input_adapter_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter_get_character = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter_get_elements = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter_m_file = R"doc(the file pointer to read from)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_file_input_adapter_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_7 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_8 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_9 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_10 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_11 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_12 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_13 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_14 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_15 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_16 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_17 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_18 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_19 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_20 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_21 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_22 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_23 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_array_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_array_impl_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_array_impl_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_array_impl_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_fn = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_fn_operator_call = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_inplace_array_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_tuple_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_tuple_impl_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_tuple_impl_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_tuple_impl_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_tuple_impl_base = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_from_json_tuple_impl_base_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_get = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_get_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_get_arithmetic_value = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_has_from_json = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_has_key_compare = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_has_non_default_from_json = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_has_to_json = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_hash =
R"doc(! hash a JSON value

The hash function tries to rely on std::hash where possible.
Furthermore, the type of the JSON value is taken into account to have
different hash values for null, 0, 0U, and false, etc.

Template parameter ``BasicJsonType``:
    basic_json specialization

Parameter ``j``:
    JSON value to hash

Returns:
    hash value of j)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_identity_tag = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_impl_is_c_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_impl_is_transparent = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_adapter_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_adapter_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_adapter_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_adapter_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_adapter_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_adapter_7 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_format_t = R"doc(the supported input formats)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_format_t_bjdata = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_format_t_bson = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_format_t_cbor = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_format_t_json = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_format_t_msgpack = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_format_t_ubjson = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter =
R"doc(! Input adapter for a (caching) istream. Ignores a UFT Byte Order Mark
at beginning of input. Does not support changing the underlying
std::streambuf in mid-input. Maintains underlying std::istream and
std::streambuf to support subsequent use of standard std::istream
operations to process any input characters following those used in
parsing the JSON input. Clears the std::istream flags; any input
errors (e.g., EOF) will be detected by the first subsequent call for
input from the std::istream.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_get_character = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_get_elements = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_input_stream_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_input_stream_adapter_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_input_stream_adapter_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_is = R"doc(the associated input stream)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_input_stream_adapter_sb = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_int_to_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_internal_iterator =
R"doc(! an iterator value

@note This structure could easily be a union, but MSVC currently does
not allow unions members with complex constructors, see
https://github.com/nlohmann/json/pull/105.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_internal_iterator_array_iterator = R"doc(iterator for JSON arrays)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_internal_iterator_object_iterator = R"doc(iterator for JSON objects)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_internal_iterator_primitive_iterator = R"doc(generic iterator for all other types)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_invalid_iterator =
R"doc(exception indicating errors with iterators

See also:
    https://json.nlohmann.me/api/basic_json/invalid_iterator/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_invalid_iterator_create = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_invalid_iterator_invalid_iterator = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_basic_json = R"doc(//////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_basic_json_context = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_c_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_comparable = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_array_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_array_type_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_integer_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_integer_type_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_object_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_object_type_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_string_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_compatible_type_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_complete_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_constructible = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_constructible_array_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_constructible_array_type_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_constructible_object_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_constructible_object_type_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_constructible_string_type = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_constructible_tuple = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_default_constructible = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_detected_lazy = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_getable = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_iterator_of_multibyte = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_iterator_of_multibyte_unnamed_enum_at_cgadimpl_include_json_hpp_6852_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_iterator_of_multibyte_unnamed_enum_at_cgadimpl_include_json_hpp_6852_5_value = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_iterator_traits = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_json_iterator_of = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_json_ref = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_ordered_map = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_ordered_map_test = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_ordered_map_test_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_ordered_map_two = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_ordered_map_two_x = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_ordered_map_unnamed_enum_at_cgadimpl_include_json_hpp_4172_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_ordered_map_unnamed_enum_at_cgadimpl_include_json_hpp_4172_5_value = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_range = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_sax = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_sax_static_asserts = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_specialization_of = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_is_transparent = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl =
R"doc(! a template for a bidirectional iterator for the basic_json class
This class implements a both iterators (iterator and const_iterator)
for the basic_json class. @note An iterator is called *initialized*
when a pointer to a JSON value has been set (e.g., by a constructor or
a copy assignment). If the iterator is default-constructed, it is
*uninitialized* and most methods are undefined. *The library uses
assertions to detect calls on uninitialized iterators.** @requirement
The class satisfies the following concept requirements: - [Bidirection
alIterator](https://en.cppreference.com/w/cpp/named_req/BidirectionalI
terator): The iterator that can be moved can be moved in both
directions (i.e. incremented and decremented). .. versionadded::
version 1.0.0, simplified in version 2.0.9, change to bidirectional
iterators in version 3.0.0 (see
https://github.com/nlohmann/json/issues/593))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_iter_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_iter_impl_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_iter_impl_3 =
R"doc(! constructor for a given JSON instance

Parameter ``object``:
    pointer to a JSON object for this iterator @pre object != nullptr
    @post The iterator is initialized; i.e. `m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_iter_impl_4 =
R"doc(! const copy constructor

Parameter ``other``:
    const iterator to copy from @note This copy constructor had to be
    defined explicitly to circumvent a bug occurring on msvc v19.0
    compiler (VS 2015) debug build. For more information refer to:
    https://github.com/nlohmann/json/issues/1608)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_iter_impl_5 =
R"doc(! converting constructor

Parameter ``other``:
    non-const iterator to copy from @note It is not checked whether
    *other* is initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_key =
R"doc(! return the key of an object iterator @pre The iterator is
initialized; i.e. `m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_m_it = R"doc(the actual iterator of the associated instance)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_m_object = R"doc(associated JSON instance)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_add =
R"doc(! add to iterator @pre The iterator is initialized; i.e. `m_object !=
nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_array =
R"doc(! access to successor @pre The iterator is initialized; i.e. `m_object
!= nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_assign_2 =
R"doc(! converting assignment

Parameter ``other``:
    const iterator to copy from

Returns:
    const/non-const iterator @note It is not checked whether *other*
    is initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_assign_3 =
R"doc(! converting assignment

Parameter ``other``:
    non-const iterator to copy from

Returns:
    const/non-const iterator @note It is not checked whether *other*
    is initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_dec =
R"doc(! post-decrement (it--) @pre The iterator is initialized; i.e.
`m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_dec_2 =
R"doc(! pre-decrement (--it) @pre The iterator is initialized; i.e.
`m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_eq =
R"doc(! comparison: equal @pre (1) Both iterators are initialized to point
to the same object, or (2) both iterators are value-initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_ge =
R"doc(! comparison: greater than or equal @pre (1) The iterator is
initialized; i.e. `m_object != nullptr`, or (2) both iterators are
value-initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_gt =
R"doc(! comparison: greater than @pre (1) Both iterators are initialized to
point to the same object, or (2) both iterators are value-initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_iadd =
R"doc(! add to iterator @pre The iterator is initialized; i.e. `m_object !=
nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_inc =
R"doc(! post-increment (it++) @pre The iterator is initialized; i.e.
`m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_inc_2 =
R"doc(! pre-increment (++it) @pre The iterator is initialized; i.e.
`m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_isub =
R"doc(! subtract from iterator @pre The iterator is initialized; i.e.
`m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_le =
R"doc(! comparison: less than or equal @pre (1) Both iterators are
initialized to point to the same object, or (2) both iterators are
value-initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_lt =
R"doc(! comparison: smaller @pre (1) Both iterators are initialized to point
to the same object, or (2) both iterators are value-initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_mul =
R"doc(! return a reference to the value pointed to by the iterator @pre The
iterator is initialized; i.e. `m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_ne =
R"doc(! comparison: not equal @pre (1) Both iterators are initialized to
point to the same object, or (2) both iterators are value-initialized.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_sub =
R"doc(! dereference the iterator @pre The iterator is initialized; i.e.
`m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_sub_2 =
R"doc(! subtract from iterator @pre The iterator is initialized; i.e.
`m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_operator_sub_3 =
R"doc(! return difference @pre The iterator is initialized; i.e. `m_object
!= nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_set_begin =
R"doc(! set the iterator to the first value @pre The iterator is
initialized; i.e. `m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_set_end =
R"doc(! set the iterator past the last value @pre The iterator is
initialized; i.e. `m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iter_impl_value =
R"doc(! return the value of an iterator @pre The iterator is initialized;
i.e. `m_object != nullptr`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy = R"doc(proxy class for the items() function)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_2 = R"doc(proxy class for the items() function)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_begin = R"doc(return iterator begin (needed for range-based for))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_container = R"doc(the container to iterate)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_end = R"doc(return iterator end (needed for range-based for))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_iteration_proxy = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_iteration_proxy_2 = R"doc(construct iteration proxy from a container)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_iteration_proxy_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_iteration_proxy_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_anchor = R"doc(the iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_array_index = R"doc(an index for arrays (used to create key names))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_array_index_last = R"doc(last stringified array index)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_array_index_str = R"doc(a string representation of the array index)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_empty_str = R"doc(an empty string (to return a reference for primitive values))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_iteration_proxy_value = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_iteration_proxy_value_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_iteration_proxy_value_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_iteration_proxy_value_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_key = R"doc(return key of the iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_operator_eq = R"doc(equality operator (needed for InputIterator))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_operator_inc = R"doc(increment operator (needed for range-based for))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_operator_inc_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_operator_mul = R"doc(dereference operator (needed for range-based for))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_operator_ne = R"doc(inequality operator (needed for range-based for))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iteration_proxy_value_value = R"doc(return value of the iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter_current = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter_empty = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter_end = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter_factory = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter_factory_create = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter_get_character = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter_get_elements = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_input_adapter_iterator_input_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_traits = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_iterator_types = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_default_base =
R"doc(! Default base class of the basic_json class.

So that the correct implementations of the copy / move ctors / assign
operators of basic_json do not require complex case distinctions (no
base class / custom base class used as customization point),
basic_json always has a base class. By default, this class is used
because it is empty and thus has no effect on the behavior of
basic_json.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref = R"doc(///////////////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_2 = R"doc(///////////////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_json_ref = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_json_ref_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_json_ref_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_json_ref_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_json_ref_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_json_ref_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_moved_or_copied = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_operator_mul = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_operator_sub = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_owned_value = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_ref_value_ref = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator =
R"doc(! a template for a reverse iterator class

Template parameter ``Base``:
    the base iterator type to reverse. Valid types are iterator (to
    create reverse_iterator) and const_iterator (to create
    const_reverse_iterator).

@requirement The class satisfies the following concept requirements: -
[BidirectionalIterator](https://en.cppreference.com/w/cpp/named_req/Bi
directionalIterator): The iterator that can be moved can be moved in
both directions (i.e. incremented and decremented). - [OutputIterator]
(https://en.cppreference.com/w/cpp/named_req/OutputIterator): It is
possible to write to the pointed-to element (only if *Base* is
iterator).

.. versionadded:: version 1.0.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_json_reverse_iterator = R"doc(create reverse iterator from iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_json_reverse_iterator_2 = R"doc(create reverse iterator from base class)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_key = R"doc(return the key of an object iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_add = R"doc(add to iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_array = R"doc(access to successor)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_dec = R"doc(post-decrement (it--))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_dec_2 = R"doc(pre-decrement (--it))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_iadd = R"doc(add to iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_inc = R"doc(post-increment (it++))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_inc_2 = R"doc(pre-increment (++it))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_sub = R"doc(subtract from iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_operator_sub_2 = R"doc(return difference)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_reverse_iterator_value = R"doc(return the value of an iterator)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_binary = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_boolean = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_end_array = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_end_object = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_key = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_null = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_number_float = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_number_integer = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_number_unsigned = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_parse_error = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_start_array = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_start_object = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_acceptor_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_allow_exceptions = R"doc(whether to throw exceptions in case of errors)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_binary = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_boolean = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_callback = R"doc(callback function)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_discarded = R"doc(a discarded value for the callback)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_end_array = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_end_object = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_errored = R"doc(whether a syntax error occurred)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_handle_value =
R"doc(!

Parameter ``v``:
    value to add to the JSON value we build during parsing

Parameter ``skip_callback``:
    whether we should skip calling the callback function; this is
    required after start_array() and start_object() SAX events,
    because otherwise we would call the callback function with an
    empty array or object, respectively.

@invariant If the ref stack is empty, then the passed value will be
the new root. @invariant If the ref stack contains a value, then it is
an array or an object to which we can add elements

Returns:
    pair of boolean (whether value should be kept) and pointer (to the
    passed value in the ref_stack hierarchy; nullptr if not kept))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_is_errored = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_json_sax_dom_callback_parser = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_json_sax_dom_callback_parser_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_json_sax_dom_callback_parser_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_keep_stack = R"doc(stack to manage which values to keep)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_key = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_key_keep_stack = R"doc(stack to manage which object keys to keep)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_m_lexer_ref = R"doc(the lexer reference to obtain the current position)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_null = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_number_float = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_number_integer = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_number_unsigned = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_object_element = R"doc(helper to hold the reference for the next object element)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_parse_error = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_ref_stack = R"doc(stack to model hierarchy of values)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_root = R"doc(the parsed JSON value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_start_array = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_start_object = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_callback_parser_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser =
R"doc(! SAX implementation to create a JSON value from SAX events

This class implements the json_sax interface and processes the SAX
events to create a JSON value which makes it basically a DOM parser.
The structure or hierarchy of the JSON value is managed by the stack
`ref_stack` which contains a pointer to the respective array or object
for each recursion depth.

After successful parsing, the value that is passed by reference to the
constructor contains the parsed value.

Template parameter ``BasicJsonType``:
    the JSON type)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_allow_exceptions = R"doc(whether to throw exceptions in case of errors)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_binary = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_boolean = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_end_array = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_end_object = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_errored = R"doc(whether a syntax error occurred)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_handle_value =
R"doc(! @invariant If the ref stack is empty, then the passed value will be
the new root. @invariant If the ref stack contains a value, then it is
an array or an object to which we can add elements)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_is_errored = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_json_sax_dom_parser =
R"doc(!

Parameter ``r``:
    reference to a JSON value that is manipulated while parsing

Parameter ``allow_exceptions_``:
    whether parse errors yield exceptions)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_json_sax_dom_parser_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_json_sax_dom_parser_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_key = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_m_lexer_ref = R"doc(the lexer reference to obtain the current position)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_null = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_number_float = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_number_integer = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_number_unsigned = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_object_element = R"doc(helper to hold the reference for the next object element)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_parse_error = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_ref_stack = R"doc(stack to model hierarchy of values)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_root = R"doc(the parsed JSON value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_start_array = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_start_object = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_json_sax_dom_parser_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer =
R"doc(! lexical analysis

This class organizes the lexical analysis during JSON deserialization.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_add = R"doc(add a character to token_buffer)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base = R"doc(////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type = R"doc(token types for the parser)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_begin_array = R"doc(< the character for array begin `[`)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_begin_object = R"doc(< the character for object begin `{`)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_end_array = R"doc(< the character for array end `]`)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_end_object = R"doc(< the character for object end `}`)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_end_of_input = R"doc(< indicating the end of the input buffer)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_literal_false = R"doc(< the `false` literal)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_literal_null = R"doc(< the `null` literal)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_literal_or_value = R"doc(< a literal or the begin of a value (only for diagnostics))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_literal_true = R"doc(< the `true` literal)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_name = R"doc(return name of values of type token_type (only used for errors))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_name_separator = R"doc(< the name separator `:`)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_parse_error = R"doc(< indicating a parse error)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_uninitialized = R"doc(< indicating the scanner is uninitialized)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_value_float = R"doc(< an floating point number -- use get_number_float() for actual value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_value_integer = R"doc(< a signed integer -- use get_number_integer() for actual value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_value_separator = R"doc(< the value separator `,`)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_value_string = R"doc(< a string -- use get_string() for actual value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_base_token_type_value_unsigned = R"doc(< an unsigned integer -- use get_number_unsigned() for actual value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_current = R"doc(the current character)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_decimal_point_char = R"doc(the decimal point)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_decimal_point_position = R"doc(the position of the decimal point in the input)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_error_message = R"doc(a description of occurred lexer errors)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_codepoint =
R"doc(! get codepoint from 4 hex characters following `\u`

For input "\u c1 c2 c3 c4" the codepoint is: (c1 * 0x1000) + (c2 *
0x0100) + (c3 * 0x0010) + c4 = (c1 << 12) + (c2 << 8) + (c3 << 4) +
(c4 << 0)

Furthermore, the possible characters '0'..'9', 'A'..'F', and 'a'..'f'
must be converted to the integers 0x0..0x9, 0xA..0xF, 0xA..0xF, resp.
The conversion is done by subtracting the offset (0x30, 0x37, and
0x57) between the ASCII value of the character and the desired integer
value.

Returns:
    codepoint (0x0000..0xFFFF) or -1 in case of an error (e.g. EOF or
    non-hex character))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_decimal_point = R"doc(return the locale-dependent decimal point)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_error_message = R"doc(return syntax error message)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_number_float = R"doc(return floating-point value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_number_integer = R"doc(return integer value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_number_unsigned = R"doc(return unsigned integer value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_position = R"doc(return position of last read token)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_string =
R"doc(return current string value (implicitly resets the token; useful only
once))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_get_token_string =
R"doc(return the last read token (for errors only). Will never contain EOF
(an arbitrary value that is not a valid char value, often -1), because
255 may legitimately occur. May contain NUL, which should be escaped.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_ia = R"doc(input adapter)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_ignore_comments =
R"doc(whether comments should be ignored (true) or signaled as errors
(false))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_lexer = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_lexer_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_lexer_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_next_byte_in_range =
R"doc(! check if the next byte(s) are inside a given range

Adds the current byte and, for each passed range, reads a new byte and
checks if it is inside the range. If a violation was detected, set up
an error message and return false. Otherwise, return true.

Parameter ``ranges``:
    list of integers; interpreted as list of pairs of inclusive lower
    and upper bound, respectively

@pre The passed list *ranges* must have 2, 4, or 6 elements; that is,
1, 2, or 3 pairs. This precondition is enforced by an assertion.

Returns:
    true if and only if no range violation was detected)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_next_unget = R"doc(whether the next get() call should just return current)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_position = R"doc(the start position of the current token)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_reset = R"doc(reset token_buffer; current character is beginning of token)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_scan = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_scan_comment =
R"doc(! scan a comment

Returns:
    whether comment could be scanned successfully)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_scan_literal =
R"doc(!

Parameter ``literal_text``:
    the literal text to expect

Parameter ``length``:
    the length of the passed literal text

Parameter ``return_type``:
    the token type to return on success)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_scan_number =
R"doc(! scan a number literal

This function scans a string according to Sect. 6 of RFC 8259.

The function is realized with a deterministic finite state machine
derived from the grammar described in RFC 8259. Starting in state
"init", the input is read and used to determined the next state. Only
state "done" accepts the number. State "error" is a trap state to
model errors. In the table below, "anything" means any character but
the ones listed before.

state | 0 | 1-9 | e E | + | - | . | anything ---------|----------|----
------|----------|---------|---------|----------|----------- init |
zero | any1 | [error] | [error] | minus | [error] | [error] minus |
zero | any1 | [error] | [error] | [error] | [error] | [error] zero |
done | done | exponent | done | done | decimal1 | done any1 | any1 |
any1 | exponent | done | done | decimal1 | done decimal1 | decimal2 |
decimal2 | [error] | [error] | [error] | [error] | [error] decimal2 |
decimal2 | decimal2 | exponent | done | done | done | done exponent |
any2 | any2 | [error] | sign | sign | [error] | [error] sign | any2 |
any2 | [error] | [error] | [error] | [error] | [error] any2 | any2 |
any2 | done | done | done | done | done

The state machine is realized with one label per state (prefixed with
"scan_number_") and `goto` statements between them. The state machine
contains cycles, but any cycle can be left when EOF is read.
Therefore, the function is guaranteed to terminate.

During scanning, the read bytes are stored in token_buffer. This
string is then converted to a signed integer, an unsigned integer, or
a floating-point number.

Returns:
    token_type::value_unsigned, token_type::value_integer, or
    token_type::value_float if number could be successfully scanned,
    token_type::parse_error otherwise

@note The scanner is independent of the current locale. Internally,
the locale's decimal point is used instead of `.` to work with the
locale-dependent converters.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_scan_string =
R"doc(! scan a string literal

This function scans a string according to Sect. 7 of RFC 8259. While
scanning, bytes are escaped and copied into buffer token_buffer. Then
the function returns successfully, token_buffer is *not* null-
terminated (as it may contain \0 bytes), and token_buffer.size() is
the number of bytes in the string.

Returns:
    token_type::value_string if string could be successfully scanned,
    token_type::parse_error otherwise

@note In case of errors, variable error_message contains a textual
description.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_skip_bom =
R"doc(! skip the UTF-8 byte order mark

Returns:
    true iff there is no BOM or the correct BOM has been skipped)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_skip_whitespace = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_strtof = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_strtof_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_strtof_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_token_buffer = R"doc(buffer for variable-length tokens (numbers, strings))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_token_string = R"doc(raw input token string (for error messages))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_unget =
R"doc(! unget current character (read it again on next get)

We implement unget by setting variable next_unget to true. The input
is not changed - we just simulate ungetting by modifying
chars_read_total, chars_read_current_line, and token_string. The next
call to get() will behave as if the unget character is read again.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_value_float = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_value_integer = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_lexer_value_unsigned = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_little_endianness =
R"doc(! determine system byte order

Returns:
    true if and only if system's byte order is little endian

@note from https://stackoverflow.com/a/1001328/266378)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_make_array = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_make_void = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_negation = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_nonesuch = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_nonesuch_nonesuch = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_nonesuch_nonesuch_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_nonesuch_nonesuch_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_nonesuch_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_nonesuch_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_operator_lt = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_other_error =
R"doc(exception indicating other library errors

See also:
    https://json.nlohmann.me/api/basic_json/other_error/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_other_error_create = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_other_error_other_error = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_out_of_range =
R"doc(exception indicating access out of the defined range

See also:
    https://json.nlohmann.me/api/basic_json/out_of_range/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_out_of_range_create = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_out_of_range_out_of_range = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_oa = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_operator_shared_ptr = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_output_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_output_adapter_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_output_adapter_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_protocol = R"doc(abstract output adapter interface)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_protocol_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_protocol_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_protocol_output_adapter_protocol = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_protocol_output_adapter_protocol_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_protocol_output_adapter_protocol_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_protocol_write_character = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_adapter_protocol_write_characters = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_stream_adapter = R"doc(output adapter for output streams)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_stream_adapter_output_stream_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_stream_adapter_stream = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_stream_adapter_write_character = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_stream_adapter_write_characters = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_string_adapter = R"doc(output adapter for basic_string)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_string_adapter_output_string_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_string_adapter_str = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_string_adapter_write_character = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_string_adapter_write_characters = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_vector_adapter = R"doc(output adapter for byte vectors)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_vector_adapter_output_vector_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_vector_adapter_v = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_vector_adapter_write_character = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_output_vector_adapter_write_characters = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_error =
R"doc(exception indicating a parse error

See also:
    https://json.nlohmann.me/api/basic_json/parse_error/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_error_byte =
R"doc(! byte index of the parse error

The byte index of the last read character in the input file.

@note For an input with n bytes, 1 is the index of the first character
and n+1 is the index of the terminating null byte or the end of file.
This also holds true when reading a byte vector (CBOR or MessagePack).)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_error_create =
R"doc(! create a parse error exception

Parameter ``id_``:
    the id of the exception

Parameter ``pos``:
    the position where the error occurred (or with chars_read_total=0
    if the position cannot be determined)

Parameter ``what_arg``:
    the explanatory string

Returns:
    parse_error object)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_error_create_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_error_parse_error = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_error_position_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_event_t = R"doc(/////////)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_event_t_array_end = R"doc(the parser read `]` and finished processing a JSON array)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_event_t_array_start = R"doc(the parser read `[` and started to process a JSON array)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_event_t_key = R"doc(the parser read a key of a value in an object)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_event_t_object_end = R"doc(the parser read `}` and finished processing a JSON object)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_event_t_object_start = R"doc(the parser read `{` and started to process a JSON object)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parse_event_t_value = R"doc(the parser finished reading a JSON value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser =
R"doc(! syntax analysis

This class implements a recursive descent parser.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_accept =
R"doc(! public accept interface

Parameter ``strict``:
    whether to expect the last token to be EOF

Returns:
    whether the input is a proper JSON text)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_allow_exceptions = R"doc(whether to throw exceptions in case of errors)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_callback = R"doc(callback function)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_exception_message = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_get_token = R"doc(get next token from lexer)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_last_token = R"doc(the type of the last read token)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_m_lexer = R"doc(the lexer)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_parse =
R"doc(! public parser interface

Parameter ``strict``:
    whether to expect the last token to be EOF

Parameter ``result``:
    parsed JSON value

Throws:
    parse_error.101 in case of an unexpected token

Throws:
    parse_error.102 if to_unicode fails or surrogate error

Throws:
    parse_error.103 if to_unicode fails)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_parser = R"doc(a parser reading from an input adapter)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_sax_parse = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_parser_sax_parse_internal = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_position_t = R"doc(struct to capture the start position of the current token)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_position_t_chars_read_current_line = R"doc(the number of characters read in the current line)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_position_t_chars_read_total = R"doc(the total number of characters read)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_position_t_lines_read = R"doc(the number of lines read)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_position_t_operator_unsigned_long = R"doc(conversion to size_t to preserve SAX interface)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_get_value = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_is_begin = R"doc(return whether the iterator can be dereferenced)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_is_end = R"doc(return whether the iterator is at end)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_m_it = R"doc(iterator as signed integer type)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_operator_add = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_operator_dec = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_operator_dec_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_operator_iadd = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_operator_inc = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_operator_inc_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_operator_isub = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_set_begin = R"doc(set iterator to a defined beginning)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_primitive_iterator_t_set_end = R"doc(set iterator to a defined past the end)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_priority_tag = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_priority_tag_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_replace_substring =
R"doc(! replace all occurrences of a substring by another string

Parameter ``s``:
    the string to manipulate; changed so that all occurrences of *f*
    are replaced with *t*

Parameter ``f``:
    the substring to replace with *t*

Parameter ``t``:
    the string to replace *f*

@pre The search string *f* must not be empty. **This precondition is
enforced with an assertion.**

.. versionadded:: version 2.0.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_count_digits =
R"doc(! count digits

Count the number of decimal (base 10) digits for an input unsigned
integer.

Parameter ``x``:
    unsigned integer number to count its digits

Returns:
    number of decimal digits)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_decimal_point = R"doc(the locale's decimal point character)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_decode =
R"doc(! check whether a string is UTF-8 encoded

The function checks each byte of a string whether it is UTF-8 encoded.
The result of the check is stored in the *state* parameter. The
function must be called initially with state 0 (accept). State 1 means
the string must be rejected, because the current byte is not allowed.
If the string is completely processed, but the state is non-zero, the
string ended prematurely; that is, the last byte indicated more bytes
should have followed.

Parameter ``state``:
    the state of the decoding

Parameter ``codep``:
    codepoint (valid only if resulting state is UTF8_ACCEPT)

Parameter ``byte``:
    next byte to decode

Returns:
    new state

@note The function has been edited: a std::array is used.

Copyright:
    Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>

See also:
    http://bjoern.hoehrmann.de/utf-8/decoder/dfa/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_dump =
R"doc(! internal implementation of the serialization function

This function is called by the public member function dump and
organizes the serialization internally. The indentation level is
propagated as additional parameter. In case of arrays and objects, the
function is called recursively.

- strings and object keys are escaped using `escape_string()` -
integer numbers are converted implicitly via `operator<<` - floating-
point numbers are converted to a string using `"%g"` format - binary
values are serialized as objects containing the subtype and the byte
array

Parameter ``val``:
    value to serialize

Parameter ``pretty_print``:
    whether the output shall be pretty-printed

Parameter ``ensure_ascii``:
    If *ensure_ascii* is true, all non-ASCII characters in the output
    are escaped with `\uXXXX` sequences, and the result consists of
    ASCII characters only.

Parameter ``indent_step``:
    the indent level

Parameter ``current_indent``:
    the current indent level (only used internally))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_dump_escaped =
R"doc(! dump escaped string

Escape a string by replacing certain special characters by a sequence
of an escape character (backslash) and another character and other
control characters by a sequence of "\u" followed by a four-digit hex
representation. The escaped string is written to output stream *o*.

Parameter ``s``:
    the string to escape

Parameter ``ensure_ascii``:
    whether to escape non-ASCII characters with \uXXXX sequences

@complexity Linear in the length of string *s*.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_dump_float =
R"doc(! dump a floating-point number

Dump a given floating-point number to output stream *o*. Works
internally with *number_buffer*.

Parameter ``x``:
    floating-point number to dump)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_dump_float_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_dump_float_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_dump_integer =
R"doc(! dump an integer

Dump a given integer to output stream *o*. Works internally with
*number_buffer*.

Parameter ``x``:
    integer number (signed or unsigned) to dump

Template parameter ``NumberType``:
    either *number_integer_t* or *number_unsigned_t*)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_error_handler = R"doc(error_handler how to react on decoding errors)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_hex_bytes =
R"doc(! convert a byte to a uppercase hex representation

Parameter ``byte``:
    byte to represent

Returns:
    representation ("00".."FF"))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_indent_char = R"doc(the indentation character)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_indent_string = R"doc(the indentation string)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_is_negative_number = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_is_negative_number_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_loc = R"doc(the locale)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_number_buffer = R"doc(a (hopefully) large enough character buffer)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_o = R"doc(the output of the serializer)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_remove_sign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_remove_sign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_serializer =
R"doc(!

Parameter ``s``:
    output stream to serialize to

Parameter ``ichar``:
    indentation character to use

Parameter ``error_handler_``:
    how to react on decoding errors)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_serializer_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_serializer_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_string_buffer = R"doc(string buffer)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_serializer_thousands_sep = R"doc(the locale's thousand separator character)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_span_input_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_span_input_adapter_get = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_span_input_adapter_ia = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_span_input_adapter_span_input_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_span_input_adapter_span_input_adapter_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_static_const = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_chars =
R"doc(! generates a decimal representation of the floating-point number
value in [first, last).

The format of the resulting decimal representation is similar to
printf's %g format. Returns an iterator pointing past-the-end of the
decimal representation.

@note The input number must be finite, i.e. NaN's and Inf's are not
supported. @note The buffer must be large enough. @note The result is
NOT null-terminated.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_5 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_6 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_7 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_8 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_9 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_10 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_11 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_12 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_13 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_14 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_15 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_16 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_17 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_18 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_19 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_20 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_21 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_fn = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_fn_operator_call = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_tuple_impl = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_json_tuple_impl_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_to_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_type_error =
R"doc(exception indicating executing a member function with a wrong type

See also:
    https://json.nlohmann.me/api/basic_json/type_error/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_type_error_create = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_type_error_type_error = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_unescape =
R"doc(! string unescaping as described in RFC 6901 (Sect. 4)

Parameter ``s``:
    string to unescape

Returns:
    unescaped string

Note the order of escaping "~1" to "/" and "~0" to "~" is important.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_unknown_size = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_in_range_of = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_in_range_of_impl1 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_in_range_of_impl2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t =
R"doc(! the JSON type enumeration

This enumeration collects the different JSON types. It is internally
used to distinguish the stored values, and the functions
basic_json::is_null(), basic_json::is_object(),
basic_json::is_array(), basic_json::is_string(),
basic_json::is_boolean(), basic_json::is_number() (with
basic_json::is_number_integer(), basic_json::is_number_unsigned(), and
basic_json::is_number_float()), basic_json::is_discarded(),
basic_json::is_primitive(), and basic_json::is_structured() rely on
it.

@note There are three enumeration entries (number_integer,
number_unsigned, and number_float), because the library distinguishes
these three types for numbers: basic_json::number_unsigned_t is used
for unsigned integers, basic_json::number_integer_t is used for signed
integers, and basic_json::number_float_t is used for floating-point
numbers or to approximate integers which do not fit in the limits of
their respective type.

See also:
    see basic_json::basic_json(const value_t value_type) -- create a
    JSON value with the default value for a given type

.. versionadded:: version 1.0.0)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_array = R"doc(< array (ordered collection of values))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_binary = R"doc(< binary array (ordered collection of bytes))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_boolean = R"doc(< boolean value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_discarded = R"doc(< discarded by the parser callback function)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_null = R"doc(< null value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_number_float = R"doc(< number value (floating-point))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_number_integer = R"doc(< number value (signed integer))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_number_unsigned = R"doc(< number value (unsigned integer))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_object = R"doc(< object (unordered set of name/value pairs))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_value_t_string = R"doc(< string value)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter_base_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter_fill_buffer = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter_get_character = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter_get_elements = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter_utf8_bytes = R"doc(a buffer for UTF-8 bytes)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter_utf8_bytes_filled = R"doc(number of valid bytes in the utf8_codes array)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter_utf8_bytes_index = R"doc(index to the utf8_codes array for the next valid byte)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_adapter_wide_string_input_adapter = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_detail_wide_string_input_helper = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer =
R"doc(JSON Pointer defines a string syntax for identifying a specific value
within a JSON document

See also:
    https://json.nlohmann.me/api/json_pointer/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_2 =
R"doc(JSON Pointer defines a string syntax for identifying a specific value
within a JSON document

See also:
    https://json.nlohmann.me/api/json_pointer/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_array_index =
R"doc(!

Parameter ``s``:
    reference token to be converted into an array index

Returns:
    integer representation of *s*

Throws:
    parse_error.106 if an array index begins with '0'

Throws:
    parse_error.109 if an array index begins not with a digit

Throws:
    out_of_range.404 if string *s* could not be converted to an
    integer

Throws:
    out_of_range.410 if an array index exceeds size_type)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_back =
R"doc(return last reference token

See also:
    https://json.nlohmann.me/api/json_pointer/back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_contains =
R"doc(!

Throws:
    parse_error.106 if an array index begins with '0'

Throws:
    parse_error.109 if an array index was not a number)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_convert = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_convert_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_empty =
R"doc(return whether pointer points to the root document

See also:
    https://json.nlohmann.me/api/json_pointer/empty/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_flatten =
R"doc(!

Parameter ``reference_string``:
    the reference string to the current value

Parameter ``value``:
    the value to consider

Parameter ``result``:
    the result object to insert values to

@note Empty objects or arrays are flattened to `null`.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_get_and_create =
R"doc(! create and return a reference to the pointed to value

@complexity Linear in the number of reference tokens.

Throws:
    parse_error.109 if array index is not a number

Throws:
    type_error.313 if value cannot be unflattened)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_get_checked =
R"doc(!

Throws:
    parse_error.106 if an array index begins with '0'

Throws:
    parse_error.109 if an array index was not a number

Throws:
    out_of_range.402 if the array index '-' is used

Throws:
    out_of_range.404 if the JSON pointer can not be resolved)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_get_checked_2 =
R"doc(!

Throws:
    parse_error.106 if an array index begins with '0'

Throws:
    parse_error.109 if an array index was not a number

Throws:
    out_of_range.402 if the array index '-' is used

Throws:
    out_of_range.404 if the JSON pointer can not be resolved)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_get_unchecked =
R"doc(! return a reference to the pointed to value

@note This version does not throw if a value is not present, but tries
to create nested values instead. For instance, calling this function
with pointer `"/this/that"` on a null value is equivalent to calling
`operator[]("this").operator[]("that")` on that value, effectively
changing the null value to an object.

Parameter ``ptr``:
    a JSON value

Returns:
    reference to the JSON value pointed to by the JSON pointer

@complexity Linear in the length of the JSON pointer.

Throws:
    parse_error.106 if an array index begins with '0'

Throws:
    parse_error.109 if an array index was not a number

Throws:
    out_of_range.404 if the JSON pointer can not be resolved)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_get_unchecked_2 =
R"doc(! return a const reference to the pointed to value

Parameter ``ptr``:
    a JSON value

Returns:
    const reference to the JSON value pointed to by the JSON pointer

Throws:
    parse_error.106 if an array index begins with '0'

Throws:
    parse_error.109 if an array index was not a number

Throws:
    out_of_range.402 if the array index '-' is used

Throws:
    out_of_range.404 if the JSON pointer can not be resolved)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_json_pointer =
R"doc(create JSON pointer

See also:
    https://json.nlohmann.me/api/json_pointer/json_pointer/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_operator_idiv =
R"doc(append another JSON pointer at the end of this JSON pointer

See also:
    https://json.nlohmann.me/api/json_pointer/operator_slasheq/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_operator_idiv_2 =
R"doc(append an unescaped reference token at the end of this JSON pointer

See also:
    https://json.nlohmann.me/api/json_pointer/operator_slasheq/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_operator_idiv_3 =
R"doc(append an array index at the end of this JSON pointer

See also:
    https://json.nlohmann.me/api/json_pointer/operator_slasheq/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_operator_typename_string_t_helper_type =
R"doc(return a string representation of the JSON pointer

See also:
    https://json.nlohmann.me/api/json_pointer/operator_string/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_parent_pointer =
R"doc(returns the parent of this JSON pointer

See also:
    https://json.nlohmann.me/api/json_pointer/parent_pointer/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_pop_back =
R"doc(remove last reference token

See also:
    https://json.nlohmann.me/api/json_pointer/pop_back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_push_back =
R"doc(append an unescaped token at the end of the reference pointer

See also:
    https://json.nlohmann.me/api/json_pointer/push_back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_push_back_2 =
R"doc(append an unescaped token at the end of the reference pointer

See also:
    https://json.nlohmann.me/api/json_pointer/push_back/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_reference_tokens = R"doc(the reference tokens)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_split =
R"doc(! split the string input to reference tokens

@note This function is only called by the json_pointer constructor.
All exceptions below are documented there.

Throws:
    parse_error.107 if the pointer is not empty or begins with '/'

Throws:
    parse_error.108 if character '~' is not followed by '0' or '1')doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_string_t_helper = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_to_string =
R"doc(return a string representation of the JSON pointer

See also:
    https://json.nlohmann.me/api/json_pointer/to_string/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_top = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_pointer_unflatten =
R"doc(!

Parameter ``value``:
    flattened JSON

Returns:
    unflattened JSON

Throws:
    parse_error.109 if array index is not a number

Throws:
    type_error.314 if value is not an object

Throws:
    type_error.315 if object values are not primitive

Throws:
    type_error.313 if value cannot be unflattened)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax =
R"doc(! SAX interface

This class describes the SAX interface used by
nlohmann::json::sax_parse. Each function is called in different
situations while the input is parsed. The boolean return value informs
the parser whether to continue processing the input.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_binary =
R"doc(! a binary value was read

Parameter ``val``:
    binary value

Returns:
    whether parsing should proceed @note It is safe to move the passed
    binary value.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_boolean =
R"doc(! a boolean value was read

Parameter ``val``:
    boolean value

Returns:
    whether parsing should proceed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_end_array =
R"doc(! the end of an array was read

Returns:
    whether parsing should proceed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_end_object =
R"doc(! the end of an object was read

Returns:
    whether parsing should proceed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_json_sax = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_json_sax_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_json_sax_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_key =
R"doc(! an object key was read

Parameter ``val``:
    object key

Returns:
    whether parsing should proceed @note It is safe to move the passed
    string.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_null =
R"doc(! a null value was read

Returns:
    whether parsing should proceed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_number_float =
R"doc(! a floating-point number was read

Parameter ``val``:
    floating-point value

Parameter ``s``:
    raw token value

Returns:
    whether parsing should proceed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_number_integer =
R"doc(! an integer number was read

Parameter ``val``:
    integer value

Returns:
    whether parsing should proceed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_number_unsigned =
R"doc(! an unsigned integer number was read

Parameter ``val``:
    unsigned integer value

Returns:
    whether parsing should proceed)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_operator_assign = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_operator_assign_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_parse_error =
R"doc(! a parse error occurred

Parameter ``position``:
    the position in the input where the error occurs

Parameter ``last_token``:
    the last read token

Parameter ``ex``:
    an exception object describing the error

Returns:
    whether parsing should proceed (must return false))doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_start_array =
R"doc(! the beginning of an array was read

Parameter ``elements``:
    number of array elements or -1 if unknown

Returns:
    whether parsing should proceed @note binary formats may report the
    number of elements)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_start_object =
R"doc(! the beginning of an object was read

Parameter ``elements``:
    number of object elements or -1 if unknown

Returns:
    whether parsing should proceed @note binary formats may report the
    number of elements)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_json_sax_string =
R"doc(! a string value was read

Parameter ``val``:
    string value

Returns:
    whether parsing should proceed @note It is safe to move the passed
    string value.)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_literals_json_literals_operator_json = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_literals_json_literals_operator_json_pointer = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_operator_eq =
R"doc(compares two JSON pointers for equality

See also:
    https://json.nlohmann.me/api/json_pointer/operator_eq/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_operator_eq_2 =
R"doc(compares JSON pointer and string for equality

See also:
    https://json.nlohmann.me/api/json_pointer/operator_eq/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_operator_eq_3 =
R"doc(compares string and JSON pointer for equality

See also:
    https://json.nlohmann.me/api/json_pointer/operator_eq/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_operator_lt = R"doc(compares two JSON pointer for less-than)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_operator_ne =
R"doc(compares two JSON pointers for inequality

See also:
    https://json.nlohmann.me/api/json_pointer/operator_ne/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_operator_ne_2 =
R"doc(compares JSON pointer and string for inequality

See also:
    https://json.nlohmann.me/api/json_pointer/operator_ne/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_operator_ne_3 =
R"doc(compares string and JSON pointer for inequality

See also:
    https://json.nlohmann.me/api/json_pointer/operator_ne/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map =
R"doc(a minimal map-like container that preserves insertion order

See also:
    https://json.nlohmann.me/api/ordered_map/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_2 =
R"doc(a minimal map-like container that preserves insertion order

See also:
    https://json.nlohmann.me/api/ordered_map/)doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_at = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_at_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_at_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_at_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_count = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_count_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_emplace = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_emplace_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_erase = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_erase_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_erase_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_erase_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_find = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_find_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_find_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_insert = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_insert_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_insert_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_m_compare = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_operator_array = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_operator_array_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_operator_array_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_operator_array_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_ordered_map = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_ordered_map_2 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_ordered_map_3 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_ordered_map_ordered_map_4 = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_to_string = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_would_call_std_begin = R"doc()doc";

static const char *__doc_nlohmann_json_abi_v3_12_0_would_call_std_end = R"doc()doc";

static const char *__doc_std_less = R"doc()doc";

static const char *__doc_std_less_operator_call = R"doc(! compare two value_t enum values .. versionadded:: version 3.0.0)doc";

static const char *__doc_std_swap = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

