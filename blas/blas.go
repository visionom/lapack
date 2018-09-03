package blas

type SParams struct {
	FLAG float32
	H11  float32
	H21  float32
	H12  float32
	H22  float32
}

type DParams struct {
	FLAG float64
	H11  float64
	H21  float64
	H12  float64
	H23  float64
}

const (
	//TransN means TRANS = 'N'  y := alpha*A*x + beta*y.
	TransN = rune('N')

	//TransT means TRANS = 'T'  y := alpha*A**T*x + beta*y.
	TransT = rune('T')

	//TransC means TRANS = 'C'  y := alpha*A**T*x + beta*y.
	TransC = rune('C')

	// UploU means UPLO = 'U' Only the upper triangular part of A is to be referenced.
	UploU = rune('U')

	// UploL means UPLO = 'L' Only the lower triangular part of A is to be referenced.
	UploL = rune('L')

	//DiagU means DIAG = 'U' or 'u'   A is assumed to be unit triangular.
	DiagU = rune('U')

	//DiagN means DIAG = 'N' or 'n'   A is not assumed to be unit triangular.
	DiagN = rune('N')
)

type BLAS interface {

	/*
	 * === LEVEL1 ===
	 */

	// --------------
	// --- SINGLE ---
	// --------------

	// SROTG setup Givens rotation
	SROTG(a, b float32) (c, s float32)

	// SROTMG setup modified Givens rotation
	SROTMG(d1, d2, x, y float32) (rd1, rd2, rx float32, p SParams)

	// SROT apply Givens rotation
	SROT(n int, x []float32, incX int, y []float32, incY int, c, s float32) (ry []float32)

	// SROTM apply modified Givens rotation
	SROTM(n int, x []float32, incX int, y []float32, incY int, p SParams) (rx, ry []float32)

	// SSWAP swap x and y
	SSWAP(n int, x []float32, incX int, y []float32, incY int) (rx, ry []float32)

	// SSCAL x = a*x
	SSCAL(n int, alpha float32, x []float32, incX int) (rx []float32)

	// SCOPY copy x into y
	SCOPY(n int, x []float32, incX int, y []float32, incY int) (ry []float32)

	// SAXPY y = a*x + y
	SAXPY(n int, alpha float32, x []float32, incX int, y []float32, incY int) (ry []float32)

	// SDOT dot product
	SDOT(n int, x []float32, incX int, y []float32, incY int) (r float32)

	// SDSDOT dot product with extended precision accumulation
	SDSDOT(n int, alpha float32, x []float32, incX int, y []float32, incY int) (r float32)

	// SNRM2 Euclidean norm
	SNRM2(n int, x []float32, incX int) (r float32)

	// SCNRM2 Euclidean norm
	SCNRM2(n int, x []complex64, incX int) (r float32)

	// SASUM sum of absolute values
	SASUM(n int, x []float32, incX int) (r float32)

	// ISAMAX index of max abs value
	ISAMAX(n int, x []float32, incX int) (r int)

	// --------------
	// --- DOUBLE ---
	// --------------

	// DROTG setup Givens rotation
	DROTG(a, b float64) (c, s float64)

	// DROTMG setup modified Givens rotation
	DROTMG(d1, d2, x, y float64) (rd1, rd2, rx float64, p DParams)

	// DROT apply Givens rotation
	DROT(n int, x []float64, incX int, y []float64, incY int, c float64, s float64) (ry []float64)

	// DROTM apply modified Givens rotation
	DROTM(n int, x []float64, incX int, y []float64, incY int, p DParams) (rx, ry []float64)

	// DSWAP swap x and y
	DSWAP(n int, x []float64, incX int, y []float64, incY int) (rx, ry []float64)

	// DSCAL x = a*x
	DSCAL(n int, alpha float64, x []float64, incX int) (rx []float64)

	// DCOPY copy x into y
	DCOPY(n int, x []float64, incX int, y []float64, incY int) (ry []float64)

	// DAXPY y = a*x + y
	DAXPY(n int, alpha float64, x []float64, incX int, y []float64, incY int) (ry []float64)

	// DDOT dot product
	DDOT(n int, x []float64, incX int, y []float64, incY int) (r float64)

	// DSDOT dot product with extended precision accumulation
	DSDOT(n int, x []float32, incX int, y []float32, incY int) (r float64)

	// DNRM2 Euclidean norm
	DNRM2(n int, x []float64, incX int) (r float64)

	// DZNRM2 Euclidean norm
	DZNRM2(n int, x []complex128, incX int) (r float64)

	// DASUM sum of absolute values
	DASUM(n int, x []float64, incX int) (r float64)

	// IDAMAX index of max abs value
	IDAMAX(n int, x []float64, incX int) (r int)

	// ---------------
	// --- COMPLEX ---
	// ---------------

	// CROTG setup Givens rotation
	CROTG(a, b complex64) (c, s complex64)

	// CSROT apply Givens rotation
	CSROT(n int, x []complex64, incX int, y []complex64, incY int, c, s complex64) (ry []complex64)

	// CSWAP swap x and y
	CSWAP(n int, x []complex64, incX int, y []complex64, incY int) (rx, ry []complex64)

	// CSCAL x = a*x
	CSCAL(n int, alpha complex64, x []complex64, incX int) (rx []complex64)

	// CSSCAL x = a*x
	CSSCAL(n int, alpha float32, x []complex64, incX int) (rx []complex64)

	// CCOPY copy x into y
	CCOPY(n int, x []complex64, incX int, y []complex64, incY int) (ry []complex64)

	// CAXPY y = a*x + y
	CAXPY(n int, alpha complex64, x []complex64, incX int, y []complex64, incY int) (ry []complex64)

	// CDOTU dot product
	CDOTU(n int, x []complex64, incX int, y []complex64, incY int) (r complex64)

	// CDOTC dot product, conjugating the first vector
	CDOTC(n int, x []complex64, incX int, y []complex64, incY int) (r complex64)

	// SCASUM sum of absolute values
	SCASUM(n int, x []complex64, incX int) (r float32)

	// ICAMAX index of max abs value
	ICAMAX(n int, x []complex64, incX int) (r int)

	// ----------------------
	// --- DOUBLE COMPLEX ---
	// ----------------------

	// ZROTG setup Givens rotation

	// ZDROTF apply Givens rotation

	// ZSWAP swap x and y
	ZSWAP(n int, x []complex128, incX int, y []complex128, incY int) (rx []complex128, ry []complex128)

	// ZSCAL x = a*x
	ZSCAL(n int, alpha complex128, x []complex128, incX int) (rx []complex128)

	// ZDSCAL x = a*x
	ZDSCAL(n int, alpha float64, x []complex128, incX int) (rx []complex128)

	// ZCOPY copy x into y
	ZCOPY(n int, x []complex128, incX int, y []complex128, incY int) (ry []complex128)

	// ZAXPY y = a*x + y
	ZAXPY(n int, alpha complex128, x []complex128, incX int, y []complex128, incY int) (ry []complex128)

	// ZDOTU dot product
	ZDOTU(n int, x []complex128, incX int, y []complex128, incY int) (r complex128)

	// ZDOTC dot product, conjugating the first vector
	ZDOTC(n int, x []complex128, incX int, y []complex128, incY int) (r complex128)

	// DZASUM sum of absolute values
	DZASUM(n int, x []complex128, incX int) (r float64)

	// IZAMAX index of max abs value
	IZAMAX(n int, x []complex128, incX int) (r int)

	/*
	 * === LEVEL 2 ===
	 */

	// --------------
	// --- SINGLE ---
	// --------------

	// SGEMV matrix vector multiply
	SGEMV(trans int, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) (ry []float32)

	// SGBMV banded matrix vector multiply
	SGBMV(trans int, m, n, kL, kU int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) (ry []float32)

	// SSYMV symmetric matrix vector multiply
	SSYMV(uplo int, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) (ry []float32)

	// SSBMV symmetric banded matrix vector multiply
	SSBMV(uplo int, n, k int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) (ry []float32)

	// SSPMV symmetric packed matrix vector multiply
	SSPMV(uplo int, n int, alpha float32, ap []float32, x []float32, incX int, beta float32, y []float32, incY int) (ry []float32)

	// STRMV triangular matrix vector multiply
	STRMV(uplo int, trans int, diag int, n int, a []float32, lda int, x []float32, incX int) (rx []float32)

	// STBMV triangular banded matrix vector multiply
	STBMV(uplo int, trans int, diag int, n, k int, a []float32, lda int, x []float32, incX int) (rx []float32)

	// STPMV triangular packed matrix vector multiply
	STPMV(uplo int, trans int, diag int, n int, ap []float32, x []float32, incX int) (rx []float32)

	// STRSV solving triangular matrix problems
	STRSV(uplo int, trans int, diag int, n int, a []float32, lda int, x []float32, incX int) (rx []float32)

	// STBSV solving triangular banded matrix problems
	STBSV(uplo int, trans int, diag int, n, k int, a []float32, lda int, x []float32, incX int) (rx []float32)

	// STPSV solving triangular packed matrix problems
	STPSV(uplo int, trans int, diag int, n int, ap []float32, x []float32, incX int) (rx []float32)

	// SGER performs the rank 1 operation A := alpha*x*y' + A
	SGER(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) (ra []float32)

	// SSYR performs the symmetric rank 1 operation A := alpha*x*x' + A
	SSYR(uplo int, n int, alpha float32, x []float32, incX int, a []float32, lda int) (ra []float32)

	// SSPR symmetric packed rank 1 operation A := alpha*x*x' + A
	SSPR(uplo int, n int, alpha float32, x []float32, incX int, ap []float32) (ra []float32)

	// SSYR2 performs the symmetric rank 2 operation, A := alpha*x*y' + alpha*y*x' + A
	SSYR2(uplo int, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) (ra []float32)

	// SSPR2 performs the symmetric packed rank 2 operation, A := alpha*x*y' + alpha*y*x' + A
	SSPR2(uplo int, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32) (ra []float32)

	// --------------
	// --- DOUBLE ---
	// --------------

	// DGEMV matrix vector multiply
	DGEMV(trans int, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) (ry []float64)

	// DGBMV banded matrix vector multiply
	DGBMV(trans int, m, n, kL, kU int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) (ry []float64)

	// DSYMV symmetric matrix vector multiply
	DSYMV(uplo int, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) (ry []float64)

	// DSBMV symmetric banded matrix vector multiply
	DSBMV(uplo int, n, k int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) (ry []float64)

	// DSPMV symmetric packed matrix vector multiply
	DSPMV(uplo int, n int, alpha float64, ap []float64, x []float64, incX int, beta float64, y []float64, incY int) (ry []float64)

	// DTRMV triangular matrix vector multiply
	DTRMV(uplo int, trans int, diag int, n int, a []float64, lda int, x []float64, incX int) (rx []float64)

	// DTBMV triangular banded matrix vector multiply
	DTBMV(uplo int, trans int, diag int, n, k int, a []float64, lda int, x []float64, incX int) (rx []float64)

	// DTPMV triangular packed matrix vector multiply
	DTPMV(uplo int, trans int, diag int, n int, ap []float64, x []float64, incX int) (rx []float64)

	// DTRSV solving triangular matrix problems
	DTRSV(uplo int, trans int, diag int, n int, a []float64, lda int, x []float64, incX int) (rx []float64)

	// DTBSV solving triangular banded matrix problems
	DTBSV(uplo int, trans int, diag int, n, k int, a []float64, lda int, x []float64, incX int) (rx []float64)

	// DTPSV solving triangular packed matrix problems
	DTPSV(uplo int, trans int, diag int, n int, ap []float64, x []float64, incX int) (rx []float64)

	// DGER performs the rank 1 operation A := alpha*x*y' + A
	DGER(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) (ra []float64)

	// DSYR performs the symmetric rank 1 operation A := alpha*x*x' + A
	DSYR(uplo int, n int, alpha float64, x []float64, incX int, a []float64, lda int) (ra []float64)

	// DSPR symmetric packed rank 1 operation A := alpha*x*x' + A
	DSPR(uplo int, n int, alpha float64, x []float64, incX int, ap []float64) (ra []float64)

	// DSYR2 performs the symmetric rank 2 operation, A := alpha*x*y' + alpha*y*x' + A
	DSYR2(uplo int, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) (ra []float64)

	// DSPR2 performs the symmetric packed rank 2 operation, A := alpha*x*y' + alpha*y*x' + A
	DSPR2(uplo int, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64) (ra []float64)

	// ---------------
	// --- COMPLEX ---
	// ---------------

	// CGEMV matrix vector multiply
	CGEMV(trans int, m, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) (ry []complex64)

	// CGBMV banded matrix vector multiply
	CGBMV(trans int, m, n, kL, kU int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) (ry []complex64)

	// CHEMV hermitian matrix vector multiply
	CHEMV(uplo int, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) (ry []complex64)

	// CHBMV hermitian banded matrix vector multiply
	CHBMV(uplo int, n, k int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) (ry []complex64)

	// CHPMV hermitian packed matrix vector multiply
	CHPMV(uplo int, n int, alpha complex64, ap []complex64, x []complex64, incX int, beta complex64, y []complex64, incY int) (ry []complex64)

	// CTRMV triangular matrix vector multiply
	CTRMV(uplo int, trans int, d int, n int, a []complex64, lda int, x []complex64, incX int) (rx []complex64)

	// CTBMV triangular banded matrix vector multiply
	CTBMV(uplo int, trans int, diag int, n, k int, a []complex64, lda int, x []complex64, incX int) (rx []complex64)

	// CTPMV triangular packed matrix vector multiply
	CTPMV(uplo int, trans int, diag int, n int, ap []complex64, x []complex64, incX int) (rx []complex64)

	// CTRSV solving triangular matrix problems
	CTRSV(uplo int, trans int, diag int, n int, a []complex64, lda int, x []complex64, incX int) (rx []complex64)

	// CTBSV solving triangular banded matrix problems
	CTBSV(uplo int, trans int, diag int, n, k int, a []complex64, lda int, x []complex64, incX int) (rx []complex64)

	// CTPSV solving triangular packed matrix problems
	CTPSV(uplo int, trans int, diag int, n int, ap []complex64, x []complex64, incX int) (rx []complex64)

	// CGERU performs the rank 1 operation A := alpha*x*y' + A
	CGERU(m, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) (ra []complex64)

	// CGERC performs the rank 1 operation A := alpha*x*conjg( y' ) + A
	CGERC(m, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) (ra []complex64)

	// CHER hermitian rank 1 operation A := alpha*x*conjg(x') + A
	CHER(uplo int, n int, alpha float32, x []complex64, incX int, a []complex64, lda int) (ra []complex64)

	// CHPR hermitian packed rank 1 operation A := alpha*x*conjg( x' ) + A
	CHPR(uplo int, n int, alpha float32, x []complex64, incX int, a []complex64) (ra []complex64)

	// CHER2 hermitian rank 2 operation
	CHER2(uplo int, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) (ra []complex64)

	// CHPR2 hermitian packed rank 2 operation
	CHPR2(uplo int, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, ap []complex64) (ra []complex64)

	// ----------------------
	// --- DOUBLE COMPLEX ---
	// ----------------------

	// ZGEMV matrix vector multiply
	ZGEMV(trans int, m, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) (ry []complex128)

	// ZGBMV banded matrix vector multiply
	ZGBMV(trans int, m, n int, kL int, kU int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) (ry []complex128)

	// ZHEMV hermitian matrix vector multiply
	ZHEMV(uplo int, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) (ry []complex128)

	// ZHBMV hermitian banded matrix vector multiply
	ZHBMV(uplo int, n, k int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) (ry []complex128)

	// ZHPMV hermitian packed matrix vector multiply
	ZHPMV(uplo int, n int, alpha complex128, ap []complex128, x []complex128, incX int, beta complex128, y []complex128, incY int) (ry []complex128)

	// ZTRMV triangular matrix vector multiply
	ZTRMV(uplo int, trans int, d int, n int, a []complex128, lda int, x []complex128, incX int) (rx []complex128)

	// ZTBMV triangular banded matrix vector multiply
	ZTBMV(uplo int, trans int, d int, n, k int, a []complex128, lda int, x []complex128, incX int) (rx []complex128)

	// ZTPMV triangular packed matrix vector multiply
	ZTPMV(uplo int, trans int, d int, n int, ap []complex128, x []complex128, incX int) (rx []complex128)

	// ZTRSV solving triangular matrix problems
	ZTRSV(uplo int, trans int, d int, n int, a []complex128, lda int, x []complex128, incX int) (rx []complex128)

	// ZTBSV solving triangular banded matrix problems
	ZTBSV(uplo int, trans int, d int, n, k int, a []complex128, lda int, x []complex128, incX int) (rx []complex128)

	// ZTPSV solving triangular packed matrix problems
	ZTPSV(uplo int, trans int, d int, n int, ap []complex128, x []complex128, incX int) (rx []complex128)

	// ZGERU performs the rank 1 operation A := alpha*x*y' + A
	ZGERU(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) (ra []complex128)

	// ZGERC performs the rank 1 operation A := alpha*x*conjg( y' ) + A
	ZGERC(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) (ra []complex128)

	// ZHER hermitian rank 1 operation A := alpha*x*conjg(x') + A
	ZHER(uplo int, n int, alpha float64, x []complex128, incX int, a []complex128, lda int) (ra []complex128)

	// ZHPR hermitian packed rank 1 operation A := alpha*x*conjg( x' ) + A
	ZHPR(uplo int, n int, alpha float64, x []complex128, incX int, a []complex128) (ra []complex128)

	// ZHER2 hermitian rank 2 operation
	ZHER2(uplo int, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) (ra []complex128)

	// ZHPR2 hermitian packed rank 2 operation
	ZHPR2(uplo int, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, ap []complex128) (ra []complex128)

	/*
	 * === LEVEL 3 ===
	 */

	// --------------
	// --- SINGLE ---
	// --------------

	// SGEMM matrix matrix multiply
	SGEMM(transA, transB int, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) (rc []float32)

	// SSYMM symmetric matrix matrix multiply
	SSYMM(side int, uplo int, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) (rc []float32)

	// SSYRK symmetric rank-k update to a matrix
	SSYRK(uplo int, t int, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) (rc []float32)

	// SSYR2K symmetric rank-2k update to a matrix
	SSYR2K(uplo int, t int, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) (rc []float32)

	// STRMM triangular matrix matrix multiply
	STRMM(side int, uplo int, trans rune, d int, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) (rb []float32)

	// STRSM solving triangular matrix with multiple right hand sides
	STRSM(side int, uplo int, trans rune, d int, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) (rb []float32)

	// --------------
	// --- DOUBLE ---
	// --------------

	// DGEMM matrix matrix multiply
	DGEMM(trans, tB int, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) (rc []float64)

	// DSYMM symmetric matrix matrix multiply
	DSYMM(s int, uplo int, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) (rc []float64)

	// DSYRK symmetric rank-k update to a matrix
	DSYRK(uplo int, t int, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) (rc []float64)

	// DSYR2K symmetric rank-2k update to a matrix
	DSYR2K(uplo int, t int, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) (rc []float64)

	// DTRMM triangular matrix matrix multiply
	DTRMM(s int, uplo int, trans int, d int, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) (rb []float64)

	// DTRSM solving triangular matrix with multiple right hand sides
	DTRSM(s int, uplo int, trans int, d int, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) (rb []float64)

	// ---------------
	// --- COMPLEX ---
	// ---------------

	// CGEMM matrix matrix multiply
	CGEMM(trans, tB int, m, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) (rc []complex64)

	// CSYMM symmetric matrix matrix multiply
	CSYMM(side int, uplo int, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) (rc []complex64)

	// CHEMM hermitian matrix matrix multiply
	CHEMM(side int, uplo int, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) (rc []complex64)

	// CSYRK symmetric rank-k update to a matrix
	CSYRK(uplo int, t int, n, k int, alpha complex64, a []complex64, lda int, beta complex64, c []complex64, ldc int) (rc []complex64)

	// CHERK hermitian rank-k update to a matrix
	CHERK(uplo int, t int, n, k int, alpha float32, a []complex64, lda int, beta float32, c []complex64, ldc int) (rc []complex64)

	// CSYR2K symmetric rank-2k update to a matrix
	CSYR2K(uplo int, t int, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) (rc []complex64)

	// CHER2K hermitian rank-2k update to a matrix
	CHER2K(uplo int, t int, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta float32, c []complex64, ldc int) (rc []complex64)

	// CTRMM triangular matrix matrix multiply
	CTRMM(side int, uplo int, trans int, d int, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int) (rb []complex64)

	// CTRSM solving triangular matrix with multiple right hand sides
	CTRSM(side int, uplo int, trans int, d int, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int) (rb []complex64)

	// ----------------------
	// --- DOUBLE COMPLEX ---
	// ----------------------

	// ZGEMM matrix matrix multiply
	ZGEMM(trans, tB int, m, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) (rc []complex128)

	// ZSYMM symmetric matrix matrix multiply
	ZSYMM(side int, uplo int, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) (rc []complex128)

	// ZHEMM hermitian matrix matrix multiply
	ZHEMM(side int, uplo int, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) (rc []complex128)

	// ZSYRK symmetric rank-k update to a matrix
	ZSYRK(uplo int, t int, n, k int, alpha complex128, a []complex128, lda int, beta complex128, c []complex128, ldc int) (rc []complex128)

	// ZHERK hermitian rank-k update to a matrix
	ZHERK(uplo int, t int, n, k int, alpha float64, a []complex128, lda int, beta float64, c []complex128, ldc int) (rc []complex128)

	// ZSYR2K symmetric rank-2k update to a matrix
	ZSYR2K(uplo int, t int, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) (rc []complex128)

	// ZHER2K hermitian rank-2k update to a matrix
	ZHER2K(uplo int, t int, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta float64, c []complex128, ldc int) (rc []complex128)

	// ZTRMM triangular matrix matrix multiply
	ZTRMM(side int, uplo int, trans int, d int, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int) (rb []complex128)

	// ZTRSM solving triangular matrix with multiple right hand sides
	ZTRSM(side int, uplo int, trans int, d int, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int) (rb []complex128)
}
