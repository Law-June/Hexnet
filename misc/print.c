/******************************************************************************
 * print.c: Console Outputs
 ******************************************************************************
 * v0.1 - 01.09.2018
 *
 * Copyright (c) 2018 Tobias Schlosser (tobias@tobias-schlosser.net)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ******************************************************************************/


#include <stdio.h>

#include "../Hexnet.h"

#include "console.h"
#include "print.h"


void print_info() {
	puts("Hexnet version " HEXNET_VERSION " Copyright (c) " HEXNET_YEAR_S " " HEXNET_AUTHOR);
}

void print_usage() {
	puts("Usage: ./Hexnet [options]");
}

void print_help() {
	puts(SGM_FC_YELLOW "Use -h to get full help." SGM_TA_RESET);
}

void print_help_full() {
	puts(
		"-h, --help                  print options\n"
		"-i, --input       <image>   input image\n"
		"-o, --output      <image>   output image\n"
		"--s2h-rad         <radius>  hexagon radius\n"
		"--h2s-len         <length>  square side length\n"
		"--compare-s2s     <image>   compare input image to <image>\n"
		"--compare-s2h               compare input to output image\n"
		"--compare-metric  <metric>  compare metric: AE / SE / MAE / MSE / RMSE / PSNR\n"
		"-d, --display               display results\n"
		"-v, --verbose               increase verbosity");
}

