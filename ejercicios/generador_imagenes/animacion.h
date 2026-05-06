#ifndef ANIM_BITMAP_H
#define ANIM_BITMAP_H

#include <avr/pgmspace.h>

#define MI_ANIMACION_WIDTH 8
#define MI_ANIMACION_HEIGHT 8
#define MI_ANIMACION_FRAMES 16

const unsigned char mi_animacion_frame_0[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11000011, 0b,
  0b11000011, 0b,
  0b11000011, 0b,
  0b11000011, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_1[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11100111, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_2[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11001011, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_3[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11100111, 0b,
  0b11110111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_4[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11100111, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_5[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11100111, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_6[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11001011, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_7[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11100111, 0b,
  0b11110111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_8[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11100111, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_9[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11100111, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_10[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b11001011, 0b,
  0b11111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_11[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11000011, 0b,
  0b11000011, 0b,
  0b11111111, 0b,
  0b11100011, 0b,
  0b11110111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_12[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b11010011, 0b,
  0b11001111, 0b,
  0b11000011, 0b,
  0b01010011, 0b,
  0b01111111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_13[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b01010011, 0b,
  0b11000011, 0b,
  0b11100011, 0b,
  0b10000011, 0b,
  0b11011011, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_14[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b01000111, 0b,
  0b10000011, 0b,
  0b11000011, 0b,
  0b11000011, 0b,
  0b00011111, 0b,
  0b11111111, 0b,
};

const unsigned char mi_animacion_frame_15[] PROGMEM = {
  0b11111111, 0b,
  0b11111111, 0b,
  0b01000111, 0b,
  0b10000011, 0b,
  0b11000011, 0b,
  0b11010011, 0b,
  0b01111111, 0b,
  0b11111111, 0b,
};

const unsigned char* mi_animacion_frames[] = {
  mi_animacion_frame_0,
  mi_animacion_frame_1,
  mi_animacion_frame_2,
  mi_animacion_frame_3,
  mi_animacion_frame_4,
  mi_animacion_frame_5,
  mi_animacion_frame_6,
  mi_animacion_frame_7,
  mi_animacion_frame_8,
  mi_animacion_frame_9,
  mi_animacion_frame_10,
  mi_animacion_frame_11,
  mi_animacion_frame_12,
  mi_animacion_frame_13,
  mi_animacion_frame_14,
  mi_animacion_frame_15,
};

#endif
