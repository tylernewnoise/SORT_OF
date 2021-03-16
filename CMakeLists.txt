cmake_minimum_required(VERSION 3.17)
project(SORT_OF)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

find_program(CLANG_FORMAT_EXECUTABLE
        NAMES clang-format
        DOC "Path to clang-format executable"
        )

include_directories(${PROJECT_SOURCE_DIR}/include/sort_of)
file(GLOB SOURCE_FILES src/*.cpp)

if (CLANG_FORMAT_EXECUTABLE)
    MESSAGE("GEIL")
    set(CLANG_FORMAT_ARGS "${CLANG_FORMAT_EXECUTABLE};-i;-style=Google")
    set(FILES_TO_FORMAT
            ${SOURCE_FILES}
            ${PROJECT_SOURCE_DIR}/include/sort_of/sort_of.h
            )
    add_custom_target(${CMAKE_PROJECT_NAME}_format
            COMMAND ${CLANG_FORMAT_ARGS} ${FILES_TO_FORMAT})
endif()

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
add_dependencies(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}_format)

target_link_libraries(
        ${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        dlib
)