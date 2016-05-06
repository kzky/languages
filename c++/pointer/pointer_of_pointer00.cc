/*
  Pointer is a container of an address
 */

#include <stdio.h>

int main(void){

	char *figure1;
	char **figure2;

	char buf1[50]="あいうえお";
	char buf2[50]="かきくけこ";

    printf ("%s\n\n", "----- Pointer is a container of an address -----");

	//buf1のポインタを代入
    figure1=buf1;
    printf ("buf1:\t%p\n", buf1);
    printf ("buf2:\t%p\n", buf2);
    printf ("figure1:\t%p\n", figure1);
    printf ("*figure2:\t%p\n", *figure2);
    printf ("figure2:\t%p\n", figure2);
    printf ("\n");

	//ポインタfigure1のポインタをfigure2に代入
	figure2=&figure1;
    printf ("buf1:\t%p\n", buf1);
    printf ("buf2:\t%p\n", buf2);
    printf ("figure1:\t%p\n", figure1);
    printf ("*figure2:\t%p\n", *figure2);
    printf ("figure2:\t%p\n", figure2);
    printf ("\n");
    
	//ポインタのポインタfigure2が指す値にbuf2を格納
    *figure2=buf2;
    printf ("buf1:\t%p\n", buf1);
    printf ("buf2:\t%p\n", buf2);
    printf ("figure1:\t%p\n", figure1);
    printf ("*figure2:\t%p\n", *figure2);
    printf ("figure2:\t%p\n", figure2);
    printf ("\n");
    printf("%s\n",figure1);

	return 0;

}
