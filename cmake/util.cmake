function(STR_ENDSWITH str postfix IS_CORRECT)
  set(${IS_CORRECT})
  string(LENGTH ${str} str_length)
  string(LENGTH ${postfix} postfix_length)
  math(EXPR begin_pos ${str_length}-${postfix_length})
  if((${begin_pos} GREATER 0) OR (${begin_pos} EQUAL 0))
    string(SUBSTRING ${str} ${begin_pos} ${postfix_length} str_postfix)
    string(COMPARE EQUAL ${str_postfix} ${postfix} ${IS_CORRECT})
  endif()
  set(${IS_CORRECT} ${${IS_CORRECT}} PARENT_SCOPE)
endfunction()

function(ONEFLOW_SOURCE_GROUP group)
  cmake_parse_arguments(ONEFLOW_SOURCE_GROUP "" "" "GLOB;GLOB_RECURSE" ${ARGN})
  if(ONEFLOW_SOURCE_GROUP_GLOB)
    file(GLOB srcs1 ${ONEFLOW_SOURCE_GROUP_GLOB})
    source_group(${group} FILES ${srcs1})
  endif()

  if(ONEFLOW_SOURCE_GROUP_GLOB_RECURSE)
    file(GLOB_RECURSE srcs2 ${ONEFLOW_SOURCE_GROUP_GLOB_RECURSE})
    source_group(${group} FILES ${srcs2})
  endif()
endfunction()

function(remove_test_cpp file_group)
  foreach(cc ${${file_group}})
    get_filename_component(cc_name_we ${cc} NAME_WE)
    STR_ENDSWITH(${cc_name_we} _test is_test_cc)
    if(${is_test_cc})
      list(REMOVE_ITEM ${file_group} ${cc})
    endif()
  endforeach()
  set(${file_group} ${${file_group}} PARENT_SCOPE)
endfunction()

#function(FindFilesMatch found_files expr)
#  file(GLOB_RECURSE ${found_files} ${expr})
#  foreach(file_i ${${found_files}})
#    file(RELATIVE_PATH relative_path ${oneflow_src_dir} ${file_i})
#    string(SUBSTRING ${relative_path} 0 2 relative_path_prefix)
#    if(${relative_path_prefix} STREQUAL ..)
#      list(REMOVE_ITEM ${found_files} ${file_i})
#    endif()
#  endforeach()
#  set(${found_files} ${${found_files}} PARENT_SCOPE)
#endfunction()

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

function(SHOW_VARIABLES)
  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

set(_COUNTER 0)
macro(copy_files file_paths source_dir dest_dir target)
  find_program(rsync rsync)
  if (rsync)
    set(CACHE_FILELIST ${PROJECT_BINARY_DIR}/cached_filename_lists/cache_${_COUNTER})
    math(EXPR _COUNTER "${_COUNTER} + 1")
    file(WRITE ${CACHE_FILELIST} "")
    foreach(file ${file_paths})
      file(RELATIVE_PATH rel_path "${source_dir}" ${file})
      file(APPEND ${CACHE_FILELIST} ${rel_path}\n)
    endforeach()
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${rsync}
      ARGS -a --files-from=${CACHE_FILELIST} ${source_dir} ${dest_dir})
  else()
    foreach(file ${file_paths})
      file(RELATIVE_PATH rel_path "${source_dir}" ${file})
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${file}"
        "${dest_dir}/${rel_path}")
    endforeach()
  endif()
endmacro()

function(add_copy_headers_target)
  cmake_parse_arguments(
      PARSED_ARGS
      ""
      "NAME;SRC;DST;INDEX_FILE"
      "DEPS"
      ${ARGN}
  )
  if(NOT PARSED_ARGS_NAME)
      message(FATAL_ERROR "name required")
  endif(NOT PARSED_ARGS_NAME)
  if(NOT PARSED_ARGS_SRC)
      message(FATAL_ERROR "src required")
  endif(NOT PARSED_ARGS_SRC)
  if(NOT PARSED_ARGS_DST)
      message(FATAL_ERROR "dst required")
  endif(NOT PARSED_ARGS_DST)
  add_custom_target("${PARSED_ARGS_NAME}_create_header_dir"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PARSED_ARGS_DST}"
  DEPENDS ${PARSED_ARGS_DEPS})

  add_custom_target("${PARSED_ARGS_NAME}_copy_headers_to_destination" ALL DEPENDS "${PARSED_ARGS_NAME}_create_header_dir")
  file(GLOB_RECURSE headers "${PARSED_ARGS_SRC}/*.h")
  file(GLOB_RECURSE cuda_headers "${PARSED_ARGS_SRC}/*.cuh")
  file(GLOB_RECURSE hpp_headers "${PARSED_ARGS_SRC}/*.hpp")
  list(APPEND headers ${cuda_headers})
  list(APPEND headers ${hpp_headers})

  foreach(header_file ${headers})
    file(RELATIVE_PATH relative_file_path ${PARSED_ARGS_SRC} ${header_file})
    add_custom_command(TARGET "${PARSED_ARGS_NAME}_copy_headers_to_destination" PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} "${PARSED_ARGS_DST}/${relative_file_path}")
  endforeach()

  if(PARSED_ARGS_INDEX_FILE)
    file(STRINGS ${PARSED_ARGS_INDEX_FILE} inventory_headers)
  endif(PARSED_ARGS_INDEX_FILE)
  foreach(header_file ${inventory_headers})
    add_custom_command(TARGET "${PARSED_ARGS_NAME}_copy_headers_to_destination" PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PARSED_ARGS_SRC}/${header_file}" "${PARSED_ARGS_DST}/${header_file}")
  endforeach()
endfunction()
