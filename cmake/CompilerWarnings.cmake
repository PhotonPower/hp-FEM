# Strict warning set applied to all hpfem targets via hpfem_set_warnings(<target>)
function(hpfem_set_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
      -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused
      -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion -Wformat=2
      -Wimplicit-fallthrough)
    if(HPFEM_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(HPFEM_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  endif()
endfunction()
