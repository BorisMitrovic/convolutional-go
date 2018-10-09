#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

//#include "random.h"

    int sz = 5; // board size
    #define sze 5
    int debug = 0;
    int isMoveLegal(int *board,int *libs,int move, int colour, int *neighs, int sz);
    int playMove(int *board,int *libs,int move, int r,int colour, int *nLegal, int *legal, int *neighs, int sz);
    int displayAll(int *board, int *libs, int *legal, int *nLegal, int sz);
    int board[sze*sze];
    int libs[sze*sze];  // positive numbers point to parent stone, negative say liberty' count
    int legal[sze*sze];
    int legals[sze*sze];
    int nLegal, nTries;   
    int neighs[4];
    int player, move,r,res;
  
int getLegalMoves(int *moves,int nMoves, int *legals)
{

    // INITIALIZE
    res=0;    
    player = 1;
    nTries=0;
    for (int i = 0; i<sz*sz; ++i){
        board[i] = 0;
        libs[i] = 0;
        legal[i] = i;
    }
    nLegal = sz*sz;
    for (int i = 0; i<4; ++i){
        neighs[i]=0;
    }
        // PLAY TO INITIAL POSITION
    for (int i = 0; i<nMoves; i++){
        for (int j = 0; j<sz*sz; j++){
            if (legal[j] == moves[i]){
                r = j;
                break;
            }
        }
        res = playMove(board,libs,moves[i],r,player,&nLegal,legal,neighs,sz);
        player = 3-player;
        if (res !=0){
            //printf("C: Decided position passed to getLegalMoves \n"); 
            return -1;
        }
    }
    
    for (int i=0; i<sze*sze; i++){
        res = isMoveLegal(board, libs, i, player, neighs, sz);
        if (res==2) return i+1; // winning move
        legals[i]=res;
    }
    
    return 0;
    

}
    
int playout(int *moves,int nMoves)
{
    //printf("C receives input\n");
    srand(time(NULL));
    debug=0;
    
        // INITIALIZE
    res=0;    
    player = 1;
    nTries=0;
    for (int i = 0; i<sz*sz; ++i){
        board[i] = 0;
        libs[i] = 0;
        legal[i] = i;
    }
    nLegal = sz*sz;
    for (int i = 0; i<4; ++i){
        neighs[i]=0;
    }
        // PLAY TO INITIAL POSITION
    for (int i = 0; i<nMoves; i++){
        for (int j = 0; j<sz*sz; j++){
            if (legal[j] == moves[i]){
                r = j;
                break;
            }
        }
        res = playMove(board,libs,moves[i],r,player,&nLegal,legal,neighs,sz);
        player = 3-player;
        if (res !=0){
            //printf("C: Decided position passed - returning result= %d\n",res); // TODO: fix on Python side
            if (res==-1) res = nMoves%2+1;
            return 10 + res;
        }
    }
    //displayAll(board,libs,legal,&nLegal,sz); 
    //printf("C finishes initialising\n");
        // PLAYOUT
    for (int i = 0; i<sz*sz-nMoves; ++i){
        if (nLegal<1) break;
        if (nTries>=10*nLegal) break;
        r = rand()%nLegal;
        move = legal[r];
        //printf(" - before play move, i= %d", i);
        res= playMove(board,libs,move,r,player,&nLegal,legal,neighs,sz);
        //printf(" - after play move, res= %d\n",res);
        if (res>0){
            //printf("C returns result\n");
            return res; 
        }
        if (res==-2){
            debug =1;
            displayAll(board,libs,legal,&nLegal,sz);
            debug =0;
            //printf("C returns failure\n");
            return -1;
        }
        if (res<0){
            i--;
            nTries++;
            //printf("nTries =%d \n",nTries);
            continue;
        }
        
        player = 3-player;
        nTries= 0;
    }
    //printf("C returnes result\n");
    // no legal moves -> opponent is the winner
    return 3-player;

}

int main()
{
    //int nMoves = 5;
    //int moves[nMoves];
    //moves[0]=0;    moves[1]=1;    moves[2]=2; moves[3]=7;    moves[4]=9;
    
    //res = playout(moves,nMoves);
    //printf("res= %d\n",res);
    //return res;

    int playMove(int *board,int *libs,int move, int r,int colour, int *nLegal, int *legal, int *neighs, int sz);
    int display(int *board, int sz);
    int displayAll(int *board, int *libs, int *legal, int *nLegal, int sz);
    
      
    int board[sz*sz];
    int libs[sz*sz];  // positive numbers point to parent stone, negative say liberty' count
    int legal[sz*sz];
    int nLegal, nTries;   
    int neighs[4];
    int player, move,r,res,nw,ng;
    srand(time(NULL));
    
    nw=0;
    ng=1000000;  

for (int x=0;x<ng;x++){
if (x%(ng/20)==0) printf(" GAME:: %d\n",x);
    // INITIALIZE
    for (int i = 0; i<sz*sz; ++i){
        board[i] = 0;
        libs[i] = 0;
        legal[i] = i;
    }
    nLegal = sz*sz;
    for (int i = 0; i<4; ++i){
        neighs[i]=0;
    }
    
    // TEST
    if (0){
        playMove(board,libs,5,5,1,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,6,6,2,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,4,4,1,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,2,2,2,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,0,0,1,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,1,1,2,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
         
        playMove(board,libs,11,11,1,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,12,12,2,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,15,15,1,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,16,16,2,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        playMove(board,libs,10,10,1,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        return 0;
        
        playMove(board,libs,13,13,2,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        
        res=playMove(board,libs,7,7,1,&nLegal,legal,neighs,sz);
        displayAll(board,libs,legal,&nLegal,sz); 
        
        printf("Winner is: %d!\n",res);
        return 0;
    }


    // PLAYOUT
    res=0;    
    player = 1;
    nTries=0;
    for (int i = 0; i<sz*sz*2; ++i){
        if (nLegal<1) break;
        if (nTries>=10*nLegal) { 
            nTries = 0;
            if (debug) printf("  NTRIES gone for player %d\n",player);
            //debug=1; 
            displayAll(board,libs,legal,&nLegal,sz); 
            //debug=0;
            break;}
        r = rand()%nLegal;
        move = legal[r];
        res= playMove(board,libs,move,r,player,&nLegal,legal,neighs,sz);
        if (res>0){
            if (debug) printf("Winner is: %d!\n",res);
            break;
            //return res;  
        }
        if (res==-2){
            debug =1;
            displayAll(board,libs,legal,&nLegal,sz);
            debug=0;
            return -1;
        }
        if (res<0){
            i--;
            nTries++;
            continue;
        }
        
        displayAll(board,libs,legal,&nLegal,sz);    
        player = 3-player;
        nTries= 0;
    }
    if (res == 0) res = 3-player;
    if (debug) printf("Winner is: %d!   nTries=%d\n",3-player, nTries);
    if (res==1) nw++;
}

    printf("Winrate = %d\n",(100*nw)/ng);
    return 3-player;
}




int isMoveLegal(int *board,int *libs,int move, int colour, int *neighs, int sz){


    int getParent(int stone, int *libs);
    int updateLiberties(int stone, int *libs, int diff);
    int liberties(int stone, int *libs);
    int neighbours(int move, int sz, int *neighs);
    

    // move is an integer
    if (board[move] != 0){
        return -1;
    }
    
    int n = neighbours(move,sz,neighs);
    int l = 0; // count libs of that stone alone
    int f = 0; // count friendly stones connected
    for (int i=0;i<n;i++){
        if (board[neighs[i]] == 0) l++;
        if (board[neighs[i]] == colour) f++;
    }
    
    
    // check if legal
    if (l==0){
        int cons = 0;
        for (int i=0;i<n;i++){
            if (board[neighs[i]] == 3-colour){
                int poss = -liberties(neighs[i],libs);
                if (poss <2){
                    return 2; // winning
                }
                if (poss >1 && poss<5){
                    int par = getParent(neighs[i],libs);
                    cons = poss-1;
                    for (int j=0;j<n;j++){
                        if (j!=i && getParent(neighs[j],libs) == par){
                            cons--;
                        }
                    }
                    if (cons == 0){
                        return 2; // winning
                    }
                    cons = 0;
                }
            }
            
        
            if (board[neighs[i]] == colour){
                int poss = -liberties(neighs[i],libs);
                if (poss >1 && poss<5){
                    int par = getParent(neighs[i],libs);
                    cons = poss-1;
                    for (int j=0;j<n;j++){
                        if (j!=i && getParent(neighs[j],libs) == par){
                            cons--;
                        }
                    }
                    if (cons > 0) break;
                }
                if (poss >4){
                    cons =1;
                    break;
                }
            }
        }
        if (cons == 0){
            // illegal move as no liberties
            return 0;
        }
        if (cons <0){
            printf("  ERROR: cons < 0 in legality check");
        }
    }
    return 1; // move is legal
 }


int playMove(int *board,int *libs,int move, int r, int colour, int *nLegal, int *legal,int *neighs, int sz){


    int getParent(int stone, int *libs);
    int updateLiberties(int stone, int *libs, int diff);
    int liberties(int stone, int *libs);
    int neighbours(int move, int sz, int *neighs);
    
    if (*nLegal<0){
        printf("  ERROR: nLegal is negative at %d\n",*nLegal);
        return -2;
    }

    // move is an integer
    if (board[move] != 0){
        if (debug) printf("  INCORRECT: tried to play a move at %d, which is occupied\n",move);
        *nLegal=*nLegal-1;
        legal[r] = legal[*nLegal];
        return -1;
    }
    
    if(debug) printf("  Move: %d\n",move);
    
    int res;
    int n = neighbours(move,sz,neighs);
    int l = 0; // count libs of that stone alone
    int f = 0; // count friendly stones connected
    for (int i=0;i<n;i++){
        if (board[neighs[i]] == 0) l++;
        if (board[neighs[i]] == colour) f++;
    }
    
    
    // check if legal
    if (l==0){
        int cons = 0;
        for (int i=0;i<n;i++){
            if (board[neighs[i]] == 3-colour){
                int poss = -liberties(neighs[i],libs);
                if (poss <2){
                    return colour;
                }
                if (poss >1 && poss<5){
                    int par = getParent(neighs[i],libs);
                    cons = poss-1;
                    for (int j=0;j<n;j++){
                        if (j!=i && getParent(neighs[j],libs) == par){
                            cons--;
                        }
                    }
                    if (cons == 0){
                        return colour;
                    }
                    cons = 0;
                }
            }
            
        
            if (board[neighs[i]] == colour){
                int poss = -liberties(neighs[i],libs);
                if (poss >1 && poss<5){
                    int par = getParent(neighs[i],libs);
                    cons = poss-1;
                    for (int j=0;j<n;j++){
                        if (j!=i && getParent(neighs[j],libs) == par){
                            cons--;
                        }
                    }
                    if (cons > 0) break;
                }
                if (poss >4){
                    cons =1;
                    break;
                }
            }
        }
        if (cons == 0){
            // illegal move as no liberties - no move played
            //nLegal[colour]=nLegal[colour]-1;
            //legal[colour][r] = legal[colour][nLegal[colour]];
            if (debug) printf("  DEBUG: illegal move as no liberties, due to rejection sampling\n");
            return -1;
        }
        if (cons <0){
            printf("  ERROR: cons < 0 in legality check");
        }
    }
    
    // update liberties
    int par=-1;
    int npar=0;
    int nlibs=0;
    int shared = 0;
    for (int i=0;i<n;i++){
        if (board[neighs[i]] == 3-colour){
            //unfriendly stones decrease libs by 1
            res = updateLiberties(neighs[i],libs,1);
            if (res == 0){
                return colour;
            }
        } else if (board[neighs[i]] == colour){
            shared ++;
            if (par==-1){
                //first encountered friendly group
                res = updateLiberties(neighs[i],libs,-l);
                
                par = getParent(neighs[i],libs);
                libs[move] = par;
            } else if (par != getParent(neighs[i],libs)){
                // merge friendly groups
                nlibs= liberties(neighs[i],libs);
                
                npar = getParent(neighs[i],libs);
                libs[npar] = par;
                libs[par] += nlibs;
            }
        }
    }
    libs[par] += shared;
    if (shared==0){
        libs[move] = -l;
    }

    board[move]=colour;
    
    //update legal
    *nLegal=*nLegal-1;
    legal[r] = legal[*nLegal];

    return 0;
}

int liberties(int stone, int *libs){
    if (libs[stone] < 0){
        return libs[stone];
    }else if (libs[libs[stone]] <0) {
        return libs[libs[stone]];
    }else {
        libs[stone] = libs[libs[stone]];
        return liberties(libs[stone],libs);
    }
}

int updateLiberties(int stone, int *libs, int diff){
    if (libs[stone] < 0){
        libs[stone] +=diff;
        return libs[stone];
    }else if (libs[libs[stone]] <0) {
        libs[libs[stone]] +=diff;
        return libs[libs[stone]];
    }else {
        libs[stone] = libs[libs[stone]];
        return updateLiberties(libs[stone],libs,diff);
    }
}

int getParent(int stone, int *libs){
    if (libs[stone] < 0){
        return stone;
    }else if (libs[libs[stone]] <0) {
        return libs[stone];
    }else {
        libs[stone] = libs[libs[stone]];
        return getParent(libs[stone],libs);
    }
}

int neighbours(int move, int sz, int *neighs){
    // populates neighs[4]
    int n=0;
    if ((move-sz) >= 0){
        neighs[n++]=move-sz;
    }
    if ((move+sz) < sz*sz){
        neighs[n++]=move+sz;
    }
    if ((move%sz) != 0){
        neighs[n++]=move-1;
    }
    if (((move+1)%sz) != 0){
        neighs[n++]=move+1;
    }
    
    return n;
}

int display(int *board, int sz){
    for (int i = 0; i<sz; i++){
        for (int j=0; j<sz; j++){
            printf(" %d",board[i*sz+j]);
        }
        printf("\n");
    }
    printf("\n");
    
    return 0;
}



int displayAll(int *board, int *libs, int *legal, int *nLegal, int sz){
    if (debug){
    
        for (int i = 0; i<sz; i++){
            for (int j=0; j<sz; j++){
                printf(" %d",board[i*sz+j]);
            }
            printf("   ");
            for (int j=0; j<sz; j++){
                printf(" %3i",libs[i*sz+j]);
            }
            printf("   ");
            for (int j=0; j<sz; j++){
                printf(" %3i",legal[i*sz+j]);
            }
            printf("\n");
        }
        printf("                        %d \n\n",*nLegal);
    }
    return 0;
}
