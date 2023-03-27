#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <exception>
// #include <boost/thread.hpp>
#include <thread>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
// Sun CC doesn't handle boost::iterator_adaptor yet
#if !defined(__SUNPRO_CC) || (__SUNPRO_CC > 0x530)

#include <boost/generator_iterator.hpp>

#endif


#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers

struct ClassVertex {
    double degree;
    char *name;
};

char poi_file[MAX_STRING], net_poi[MAX_STRING], net_poi_reg[MAX_STRING], net_poi_time[MAX_STRING], net_poi_word[MAX_STRING];

char user_file[MAX_STRING], net_user[MAX_STRING], net_user_reg[MAX_STRING], net_user_time[MAX_STRING], net_user_word[MAX_STRING], net_user_poi[MAX_STRING];

char emb_poi[MAX_STRING], emb_reg[MAX_STRING], emb_time[MAX_STRING], emb_word[MAX_STRING], emb_user[MAX_STRING];


struct ClassVertex *vertex_poi;
struct ClassVertex *vertex_pv;
struct ClassVertex *vertex_pr;
struct ClassVertex *vertex_pt;
struct ClassVertex *vertex_pw;

struct ClassVertex *vertex_user;
struct ClassVertex *vertex_uv;
struct ClassVertex *vertex_ur;
struct ClassVertex *vertex_ut;
struct ClassVertex *vertex_uw;
struct ClassVertex *vertex_up;


int is_binary = 0, num_threads = 2, dim = 20, num_negative = 5;

int *pvertex_hash_table, *pword_hash_table, *pregion_hash_table, *ptime_hash_table, *neg_table_pv, *neg_table_pr, *neg_table_pt, *neg_table_pw;

int *uvertex_hash_table, *uword_hash_table, *uregion_hash_table, *utime_hash_table, *upoi_hash_table, *neg_table_uv, *neg_table_ur, *neg_table_ut, *neg_table_uw, *neg_table_up;

int max_num_vertices = 1000, num_vertices_poi = 0, num_vertices_pv = 0, num_vertices_pr = 0, num_vertices_pt = 0, num_vertices_pw = 0;

int num_vertices_user = 0, num_vertices_uv = 0, num_vertices_ur = 0, num_vertices_ut = 0, num_vertices_uw = 0, num_vertices_up = 0;

long long total_samples = 100, current_sample_count = 0, num_edges_pv = 0, num_edges_pr = 0, num_edges_pt = 0, num_edges_pw = 0;

long long num_edges_uv = 0, num_edges_ur = 0, num_edges_ut = 0, num_edges_uw = 0, num_edges_up = 0;

real init_rho = 0.025, rho;
real *emb_vertex_p, *emb_vertex_r, *emb_vertex_t, *emb_vertex_w, *sigmoid_table;

real *emb_vertex_u;

int *pv_edge_source_id, *pv_edge_target_id, *pr_edge_source_id, *pr_edge_target_id, *pt_edge_source_id, *pt_edge_target_id, *pw_edge_source_id, *pw_edge_target_id;
double *pv_edge_weight, *pr_edge_weight, *pt_edge_weight, *pw_edge_weight;

int *uv_edge_source_id, *uv_edge_target_id, *ur_edge_source_id, *ur_edge_target_id, *ut_edge_source_id, *ut_edge_target_id, *uw_edge_source_id, *uw_edge_target_id, *up_edge_source_id, *up_edge_target_id;
double *uv_edge_weight, *ur_edge_weight, *ut_edge_weight, *uw_edge_weight, *up_edge_weight;

// Parameters for edge sampling
long long *alias_pv, *alias_pr, *alias_pt, *alias_pw;
double *prob_pv, *prob_pr, *prob_pt, *prob_pw;

// Parameters for edge sampling
long long *alias_uv, *alias_ur, *alias_ut, *alias_uw, *alias_up;
double *prob_uv, *prob_ur, *prob_ut, *prob_uw, *prob_up;

//random generator
typedef boost::minstd_rand base_generator_type;
base_generator_type generator(42u);
boost::uniform_real<> uni_dist(0, 1);
boost::variate_generator<base_generator_type &, boost::uniform_real<> > uni(generator, uni_dist);


/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key) {
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*key) {
        hash = hash * seed + (*key++);
    }
    return hash % hash_table_size;
}

void InitHashTable() {
    pvertex_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) pvertex_hash_table[k] = -1;

    pregion_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) pregion_hash_table[k] = -1;

    ptime_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) ptime_hash_table[k] = -1;

    pword_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) pword_hash_table[k] = -1;

    uvertex_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) uvertex_hash_table[k] = -1;

    uregion_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) uregion_hash_table[k] = -1;

    utime_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) utime_hash_table[k] = -1;

    uword_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) uword_hash_table[k] = -1;

    upoi_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) upoi_hash_table[k] = -1;
}

void InsertHashTable(char *key, int value, int flag) {
    int addr = Hash(key);
    if (flag == 0) {
        while (pvertex_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        pvertex_hash_table[addr] = value;
    } else if (flag == 1) {
        while (pword_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        pword_hash_table[addr] = value;
    } else if (flag == 2) {
        while (pregion_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        pregion_hash_table[addr] = value;
    } else if (flag == 3) {
        while (ptime_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        ptime_hash_table[addr] = value;
    } else if (flag == 4) {
        while (uvertex_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        uvertex_hash_table[addr] = value;
        //printf("insert into uvertex_hash_table key:%s$ value:%d$ %d\n",key,value,addr);
    } else if (flag == 5) {
        while (uword_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        uword_hash_table[addr] = value;
    } else if (flag == 6) {
        while (uregion_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        uregion_hash_table[addr] = value;
    } else if (flag == 7) {
        while (utime_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        utime_hash_table[addr] = value;
    } else {
        while (upoi_hash_table[addr] != -1) {
            addr = (addr + 1) % hash_table_size;
        }
        upoi_hash_table[addr] = value;
    }
}

int SearchHashTable(char *key, ClassVertex *vertex, int flag) {
    int addr = Hash(key);
    //std::cout<<key<<"\n";
    if (flag == 0) {
        while (1) {
            if (pvertex_hash_table[addr] == -1) return -1;
            if (!strcmp(key, vertex[pvertex_hash_table[addr]].name)) return pvertex_hash_table[addr];
            addr = (addr + 1) % hash_table_size;
        }
    } else if (flag == 1) {
        while (1) {
            if (pword_hash_table[addr] == -1) return -1;
            if (!strcmp(key, vertex[pword_hash_table[addr]].name)) return pword_hash_table[addr];
            addr = (addr + 1) % hash_table_size;
        }
    } else if (flag == 2) {
        while (1) {
            if (pregion_hash_table[addr] == -1) return -1;
            if (!strcmp(key, vertex[pregion_hash_table[addr]].name)) return pregion_hash_table[addr];
            addr = (addr + 1) % hash_table_size;
        }
    } else if (flag == 3) {
        while (1) {
            if (ptime_hash_table[addr] == -1) return -1;
            if (!strcmp(key, vertex[ptime_hash_table[addr]].name)) return ptime_hash_table[addr];
            addr = (addr + 1) % hash_table_size;
        }
    }
    if (flag == 4) {
        while (1) {
            if (uvertex_hash_table[addr] == -1) {
                //printf("%dsearch hash table -1 key:%s& addr:%d&\n",flag,key,addr);
                return -1;
            }
            if (!strcmp(key, vertex[uvertex_hash_table[addr]].name)) return uvertex_hash_table[addr];
            //printf("%dsearch key:%s& value:%d$ vertex:%s&",flag,key,uvertex_hash_table[addr],vertex[uvertex_hash_table[addr]].name);
            addr = (addr + 1) % hash_table_size;
        }
    } else if (flag == 5) {
        while (1) {
            if (uword_hash_table[addr] == -1) return -1;
            if (!strcmp(key, vertex[uword_hash_table[addr]].name)) return uword_hash_table[addr];
            addr = (addr + 1) % hash_table_size;
        }
    } else if (flag == 6) {
        while (1) {
            if (uregion_hash_table[addr] == -1) return -1;
            if (!strcmp(key, vertex[uregion_hash_table[addr]].name)) return uregion_hash_table[addr];
            addr = (addr + 1) % hash_table_size;
        }
    } else if (flag == 7) {
        while (1) {
            if (utime_hash_table[addr] == -1) return -1;
            if (!strcmp(key, vertex[utime_hash_table[addr]].name)) return utime_hash_table[addr];
            addr = (addr + 1) % hash_table_size;
        }
    } else {
        while (1) {
            if (upoi_hash_table[addr] == -1) return -1;
            if (!strcmp(key, vertex[upoi_hash_table[addr]].name)) return upoi_hash_table[addr];
            addr = (addr + 1) % hash_table_size;
        }
    }

    return -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name, ClassVertex *&vertex, int &num_vertices, int flag) {
    int length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vertex[num_vertices].name = (char *) calloc(length, sizeof(char));
    strcpy(vertex[num_vertices].name, name);
    vertex[num_vertices].degree = 0;
    num_vertices++;
    if (num_vertices + 2 >= max_num_vertices) {
        max_num_vertices += 1000;
        vertex = (struct ClassVertex *) realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
    }
    //printf("%dinsert vertex:%d# name:%s@\n",flag,num_vertices-1,vertex[num_vertices-1].name);
    InsertHashTable(name, num_vertices - 1, flag);
    return num_vertices - 1;
}


/* Read network from the training file */
void ReadFile(char *network_file, long long &num_edges, int &num_vertices,
              int *&edge_source_id, int *&edge_target_id, double *&edge_weight, ClassVertex *&vertex,
              int flag) {
    FILE *fin;
    char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
    int vid;
    double weight;
    printf("open file %s\n", network_file);
    fin = fopen(network_file, "rb");
    if (fin == NULL) {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    num_edges = 0;
    while (fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);
    // printf("flag %d Number of edges: %lld          \n", num_edges);
    edge_source_id = (int *) malloc(num_edges * sizeof(int));
    edge_target_id = (int *) malloc(num_edges * sizeof(int));
    edge_weight = (double *) malloc(num_edges * sizeof(double));
    if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL) {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    fin = fopen(network_file, "rb");
    num_vertices = 0;
    for (int k = 0; k != num_edges; k++) {
        fscanf(fin, "%s\t%s\t%lf", name_v1, name_v2, &weight);

        /*if (k % 10000 == 0)
        {
            printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
            fflush(stdout);
        }*/
        if (flag == 0) {
            vid = SearchHashTable(name_v1, vertex, 0);
            if (vid == -1) { printf("Error: poi false point type!\n"); }
            if (vertex[vid].degree == 0) { num_vertices++; }
            vertex[vid].degree += weight;
            edge_source_id[k] = vid;
        } else if (flag == 4) {
            vid = SearchHashTable(name_v1, vertex, 4);
            if (vid == -1) {
                printf("Error: user false point type! %d name %s\n", k, name_v1);
            }
            if (vertex[vid].degree == 0) { num_vertices++; }
            vertex[vid].degree += weight;
            edge_source_id[k] = vid;
        } else if (flag < 4) {
            vid = SearchHashTable(name_v1, vertex_poi, 0);
            edge_source_id[k] = vid;
        } else {
            vid = SearchHashTable(name_v1, vertex_user, 4);
            edge_source_id[k] = vid;
        }


        vid = SearchHashTable(name_v2, vertex, flag);
        if (vid == -1) {
            vid = AddVertex(name_v2, vertex, num_vertices, flag);
        }
        if ((flag == 0||flag==4) && vertex[vid].degree == 0) { num_vertices++; }
        vertex[vid].degree += weight;
        edge_target_id[k] = vid;

        edge_weight[k] = weight;
    }
    fclose(fin);
    //printf("Number of vertices: %lld          \n", num_vertices);
}

void ReadPOIs(char *POI_file) {
    FILE *fin;
    char name[MAX_STRING], str[MAX_STRING + 10];
    int num_poi = 0, vid;
    printf("open file %s\n", POI_file);
    fin = fopen(POI_file, "rb");
    if (fin == NULL) {
        printf("ERROR: network file not found!\n");
        exit(1);
    }

    while (fgets(str, sizeof(str), fin)) num_poi++;
    printf("numpoi %d\n", num_poi);
    fclose(fin);

    fin = fopen(POI_file, "rb");
    for (int k = 0; k != num_poi; k++) {
        fscanf(fin, "%s", name);
        vid = SearchHashTable(name, vertex_poi, 0);
        if (vid == -1) {
            vid = AddVertex(name, vertex_poi, num_vertices_poi, 0);
        }
    }
    fclose(fin);
}

void ReadUsers(char *USER_file) {

    FILE *fin;
    char name[MAX_STRING], str[MAX_STRING + 10];
    int num_user = 0, vid;
    printf("open file %s\n", USER_file);

    fin = fopen(USER_file, "rb");
    if (fin == NULL) {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    while (fgets(str, sizeof(str), fin)) num_user++;
    fclose(fin);
    printf("numuser %d\n", num_user);
    fin = fopen(USER_file, "rb");
    for (int k = 0; k != num_user; k++) {
        fscanf(fin, "%s", name);
        vid = SearchHashTable(name, vertex_user, 4);
        if (vid == -1) {
            vid = AddVertex(name, vertex_user, num_vertices_user, 4);
        }
    }

    fclose(fin);
}

void ReadData() {
    char *name;
    int max_num = 1000;
    /* Init vertex_v* 's v poit in different graph */
    for (int i = 0; i < num_vertices_poi; i++) {
        if (i + 2 >= max_num) {
            max_num += 1000;
            vertex_pv = (struct ClassVertex *) realloc(vertex_pv, max_num * sizeof(struct ClassVertex));
        }
        name = vertex_poi[i].name;
        //std::cout<<name<<"\n";
        int length = strlen(name) + 1;
        if (length > MAX_STRING) length = MAX_STRING;
        vertex_pv[i].name = (char *) calloc(length, sizeof(char));
        strcpy(vertex_pv[i].name, name);
        //printf("inpv:%d$ name:%s@\n",i,vertex_pv[i].name);
        vertex_pv[i].degree = 0;
    }
    max_num = 1000;

    for (int i = 0; i < num_vertices_user; i++) {
        if (i + 2 >= max_num) {

            max_num += 1000;
            vertex_uv = (struct ClassVertex *) realloc(vertex_uv, max_num * sizeof(struct ClassVertex));
        }
        name = vertex_user[i].name;
        //std::cout<<name<<"\n";
        int length = strlen(name) + 1;
        if (length > MAX_STRING) length = MAX_STRING;
        vertex_uv[i].name = (char *) calloc(length, sizeof(char));
        strcpy(vertex_uv[i].name, name);
        //printf("inuv:%d$ name:%s@ %s*\n",i,vertex_uv[i].name,name);
        vertex_uv[i].degree = 0;
    }
    ReadFile(net_poi, num_edges_pv, num_vertices_pv, pv_edge_source_id, pv_edge_target_id, pv_edge_weight, vertex_pv, 0);
    std::cout << "Number of edges in net_POI:" << "\t";
    std::cout << num_edges_pv << "\n";
    std::cout << "Number of vertices of pvs:" << "\t";
    std::cout << num_vertices_pv << "\n";

    max_num_vertices = 1000;
    ReadFile(net_poi_word, num_edges_pw, num_vertices_pw, pw_edge_source_id, pw_edge_target_id, pw_edge_weight,
             vertex_pw, 1);
    std::cout << "Number of edges in net_POI_word:" << "\t";
    std::cout << num_edges_pw << "\n";
    std::cout << "Number of vertices of pwords:" << "\t";
    std::cout << num_vertices_pw << "\n";

    max_num_vertices = 1000;
    ReadFile(net_poi_reg, num_edges_pr, num_vertices_pr, pr_edge_source_id, pr_edge_target_id, pr_edge_weight,
             vertex_pr, 2);
    std::cout << "Number of edges in net_POI_reg:" << "\t";
    std::cout << num_edges_pr << "\n";
    std::cout << "Number of vertices of pregions:" << "\t";
    std::cout << num_vertices_pr << "\n";

    max_num_vertices = 1000;
    ReadFile(net_poi_time, num_edges_pt, num_vertices_pt, pt_edge_source_id, pt_edge_target_id, pt_edge_weight,
             vertex_pt, 3);
    std::cout << "Number of edges in net_POI_time:" << "\t";
    std::cout << num_edges_pt << "\n";
    std::cout << "Number of vertices of ptime slots:" << "\t";
    std::cout << num_vertices_pt << "\n";


    ReadFile(net_user, num_edges_uv, num_vertices_uv, uv_edge_source_id, uv_edge_target_id, uv_edge_weight, vertex_uv,
             4);
    std::cout << "Number of edges in net_user:" << "\t";
    std::cout << num_edges_uv << "\n";
    std::cout << "Number of vertices of uvs:" << "\t";
    std::cout << num_vertices_uv << "\n";


    max_num_vertices = 1000;
    ReadFile(net_user_word, num_edges_uw, num_vertices_uw, uw_edge_source_id, uw_edge_target_id, uw_edge_weight,
             vertex_uw, 5);
    std::cout << "Number of edges in net_user_word:" << "\t";
    std::cout << num_edges_uw << "\n";
    std::cout << "Number of vertices of uwords:" << "\t";
    std::cout << num_vertices_uw << "\n";

    max_num_vertices = 1000;
    ReadFile(net_user_reg, num_edges_ur, num_vertices_ur, ur_edge_source_id, ur_edge_target_id, ur_edge_weight,
             vertex_ur, 6);
    std::cout << "Number of edges in net_user_reg:" << "\t";
    std::cout << num_edges_ur << "\n";
    std::cout << "Number of vertices of uregions:" << "\t";
    std::cout << num_vertices_ur << "\n";

    max_num_vertices = 1000;
    ReadFile(net_user_time, num_edges_ut, num_vertices_ut, ut_edge_source_id, ut_edge_target_id, ut_edge_weight,
             vertex_ut, 7);
    std::cout << "Number of edges in net_user_time:" << "\t";
    std::cout << num_edges_ut << "\n";
    std::cout << "Number of vertices of utime slots:" << "\t";
    std::cout << num_vertices_ut << "\n";

    max_num_vertices = 1000;
    ReadFile(net_user_poi, num_edges_up, num_vertices_up, up_edge_source_id, up_edge_target_id, up_edge_weight,
             vertex_up, 8);
    std::cout << "Number of edges in net_user_poi:" << "\t";
    std::cout << num_edges_up << "\n";
    std::cout << "Number of vertices of upois:" << "\t";
    std::cout << num_vertices_up << "\n";


    free(pvertex_hash_table);
    free(pword_hash_table);
    free(pregion_hash_table);
    free(ptime_hash_table);

    free(uvertex_hash_table);
    free(uword_hash_table);
    free(uregion_hash_table);
    free(utime_hash_table);
    free(upoi_hash_table);
}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable(long long *&alias, double *&prob, long long num_edges, double *edge_weight) {
    alias = (long long *) malloc(num_edges * sizeof(long long));
    prob = (double *) malloc(num_edges * sizeof(double));
    if (alias == NULL || prob == NULL) {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double *norm_prob = (double *) malloc(num_edges * sizeof(double));
    long long *large_block = (long long *) malloc(num_edges * sizeof(long long));
    long long *small_block = (long long *) malloc(num_edges * sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL) {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
    for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

    for (long long k = num_edges - 1; k >= 0; k--) {
        if (norm_prob[k] < 1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block) {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob[cur_small_block] = norm_prob[cur_small_block];
        alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;

    free(norm_prob);
    free(small_block);
    free(large_block);
}

void InitAlias() {
    InitAliasTable(alias_pv, prob_pv, num_edges_pv, pv_edge_weight);
    InitAliasTable(alias_pr, prob_pr, num_edges_pr, pr_edge_weight);
    InitAliasTable(alias_pt, prob_pt, num_edges_pt, pt_edge_weight);
    InitAliasTable(alias_pw, prob_pw, num_edges_pw, pw_edge_weight);

    InitAliasTable(alias_uv, prob_uv, num_edges_uv, uv_edge_weight);
    InitAliasTable(alias_ur, prob_ur, num_edges_ur, ur_edge_weight);
    InitAliasTable(alias_ut, prob_ut, num_edges_ut, ut_edge_weight);
    InitAliasTable(alias_uw, prob_uw, num_edges_uw, uw_edge_weight);
    InitAliasTable(alias_up, prob_up, num_edges_up, up_edge_weight);

}

long long SampleAnEdge(double rand_value1, double rand_value2, int num_edges, long long *alias, double *prob) {
    long long k = (long long) num_edges * rand_value1;
    return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector() {
    long long a, b;
    //int num;
    //vertex of poi
    // emb_vertex_p = (real *)memalign( 128, (long long)num_vertices_poi * dim * sizeof(real));
    posix_memalign((void **) &emb_vertex_p, 128, (long long) num_vertices_poi * dim * sizeof(real));
    if (emb_vertex_p == NULL) {
        printf("Error: memory allocation failed\n");
        exit(1);
    }
    for (b = 0; b < dim; b++)
        for (a = 0; a < num_vertices_poi; a++)
            emb_vertex_p[a * dim + b] = (rand() / (real) RAND_MAX - 0.5) / dim;
    //emb_vertex_p[a * dim + b] = 0;

    posix_memalign((void **) &emb_vertex_u, 128, (long long) num_vertices_user * dim * sizeof(real));
    if (emb_vertex_u == NULL) {
        printf("Error: memory allocation failed\n");
        exit(1);
    }
    for (b = 0; b < dim; b++)
        for (a = 0; a < num_vertices_user; a++)
            emb_vertex_u[a * dim + b] = (rand() / (real) RAND_MAX - 0.5) / dim;

    printf("%d %d %d %d %d %d\n", num_vertices_pr, num_vertices_ur, num_vertices_pw, num_vertices_uw, num_vertices_pt,
           num_vertices_pt);
    assert(num_vertices_pr == num_vertices_ur && num_vertices_pw == num_vertices_uw &&
           num_vertices_pt == num_vertices_pt);

    //vertex of region
    // emb_vertex_r = (real *)memalign(128,(long long)num_vertices_r * dim * sizeof(real));
    posix_memalign((void **) &emb_vertex_r, 128, (long long) num_vertices_pr * dim * sizeof(real));
    if (emb_vertex_r == NULL) {
        printf("Error: memory allocation failed\n");
        exit(1);
    }
    for (b = 0; b < dim; b++)
        for (a = 0; a < num_vertices_pr; a++)
            emb_vertex_r[a * dim + b] = (rand() / (real) RAND_MAX - 0.5) / dim;
    //emb_vertex_r[a * dim + b] = 0;

    //vertex of time
    // emb_vertex_t = (real *)memalign(128,(long long)num_vertices_t * dim * sizeof(real));
    posix_memalign((void **) &emb_vertex_t, 128, (long long) num_vertices_pt * dim * sizeof(real));
    if (emb_vertex_t == NULL) {
        printf("Error: memory allocation failed\n");
        exit(1);
    }
    for (b = 0; b < dim; b++)
        for (a = 0; a < num_vertices_pt; a++)
            emb_vertex_t[a * dim + b] = (rand() / (real) RAND_MAX - 0.5) / dim;
    //emb_vertex_t[a * dim + b] = 0;

    //vertex of word
    // emb_vertex_w = (real *)memalign(128,(long long)num_vertices_w * dim * sizeof(real));
    posix_memalign((void **) &emb_vertex_w, 128, (long long) num_vertices_pw * dim * sizeof(real));
    if (emb_vertex_w == NULL) {
        printf("Error: memory allocation failed\n");
        exit(1);
    }
    for (b = 0; b < dim; b++)
        for (a = 0; a < num_vertices_pw; a++)
            emb_vertex_w[a * dim + b] = (rand() / (real) RAND_MAX - 0.5) / dim;
    //emb_vertex_w[a * dim + b] = 0;
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable(int *&neg_table, int num_vertices, ClassVertex *vertex) {
    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    neg_table = (int *) malloc(neg_table_size * sizeof(int));

    for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++) {
        if ((double) (k + 1) / neg_table_size > por) {
            cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid++;
        }
        neg_table[k] = vid - 1;
    }
}

void InitNeg() {
    InitNegTable(neg_table_pv, num_vertices_pv, vertex_pv);
    InitNegTable(neg_table_pr, num_vertices_pr, vertex_pr);
    InitNegTable(neg_table_pt, num_vertices_pt, vertex_pt);
    InitNegTable(neg_table_pw, num_vertices_pw, vertex_pw);

    InitNegTable(neg_table_uv, num_vertices_uv, vertex_uv);
    InitNegTable(neg_table_ur, num_vertices_ur, vertex_ur);
    InitNegTable(neg_table_ut, num_vertices_ut, vertex_ut);
    InitNegTable(neg_table_uw, num_vertices_uw, vertex_uw);
    InitNegTable(neg_table_up, num_vertices_up, vertex_up);
}

/* Fastly compute sigmoid function */
void InitSigmoidTable() {
    real x;
    sigmoid_table = (real *) malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++) {
        x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

real FastSigmoid(real x) {
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed) {
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label,int a) {
    real x = 0, g;

    for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
    if (isnan(x)) {
        std::cout << "错了 "<<a<<" " <<x<< std::endl;

    }
    g = (label - FastSigmoid(x)) * rho;
    for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
    for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

void *TrainLINEThread(long id) {
    long long u, v, lu, lv, target, label;
    long long count = 0, last_count = 0, curedge;
    unsigned long long seed = (long long) id;
    int *neg_table, *edge_source_id, *edge_target_id;
    real *emb_vertex_target, *emb_context_target;
    real *vec_error = (real *) calloc(dim, sizeof(real));

    while (1) {
        //judge for exit
        if (count > total_samples / num_threads + 2) break;

        if (count - last_count > 10000) {
            current_sample_count += count - last_count;
            last_count = count;
            //printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
            //fflush(stdout);
            rho = init_rho * (1 - current_sample_count / (real) (total_samples + 1));
            if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
        }

        int a = count % 9;
        //std::cout<<a<<"\n";


        switch (a) {
            case 0:
                curedge = SampleAnEdge(uni(), uni(), num_edges_pr, alias_pr, prob_pr);
                neg_table = neg_table_pr;
                emb_vertex_target = emb_vertex_r;
                emb_context_target = emb_vertex_p;
                edge_source_id = pr_edge_source_id;
                edge_target_id = pr_edge_target_id;
                break;

            case 1:
                curedge = SampleAnEdge(uni(), uni(), num_edges_pt, alias_pt, prob_pt);
                neg_table = neg_table_pt;
                emb_vertex_target = emb_vertex_t;
                emb_context_target = emb_vertex_p;
                edge_source_id = pt_edge_source_id;
                edge_target_id = pt_edge_target_id;
                break;
            case 2:
                curedge = SampleAnEdge(uni(), uni(), num_edges_pw, alias_pw, prob_pw);
                neg_table = neg_table_pw;
                emb_vertex_target = emb_vertex_w;
                emb_context_target = emb_vertex_p;
                edge_source_id = pw_edge_source_id;
                edge_target_id = pw_edge_target_id;
                break;
            case 3:
                curedge = SampleAnEdge(uni(), uni(), num_edges_pv, alias_pv, prob_pv);
                neg_table = neg_table_pv;
                emb_vertex_target = emb_vertex_p;
                emb_context_target = emb_vertex_p;
                edge_source_id = pv_edge_source_id;
                edge_target_id = pv_edge_target_id;
                break;
            case 4:
                curedge = SampleAnEdge(uni(), uni(), num_edges_ur, alias_ur, prob_ur);
                neg_table = neg_table_ur;
                emb_vertex_target = emb_vertex_r;
                emb_context_target = emb_vertex_u;
                edge_source_id = ur_edge_source_id;
                edge_target_id = ur_edge_target_id;
                break;

            case 5:
                curedge = SampleAnEdge(uni(), uni(), num_edges_ut, alias_ut, prob_ut);
                neg_table = neg_table_ut;
                emb_vertex_target = emb_vertex_t;
                emb_context_target = emb_vertex_u;
                edge_source_id = ut_edge_source_id;
                edge_target_id = ut_edge_target_id;
                break;
            case 6:
                curedge = SampleAnEdge(uni(), uni(), num_edges_uw, alias_uw, prob_uw);
                neg_table = neg_table_uw;
                emb_vertex_target = emb_vertex_w;
                emb_context_target = emb_vertex_u;
                edge_source_id = uw_edge_source_id;
                edge_target_id = uw_edge_target_id;
                break;
            case 7:
                curedge = SampleAnEdge(uni(), uni(), num_edges_uv, alias_uv, prob_uv);
                neg_table = neg_table_uv;
                emb_vertex_target = emb_vertex_u;
                emb_context_target = emb_vertex_u;
                edge_source_id = uv_edge_source_id;
                edge_target_id = uv_edge_target_id;
                break;
            default:
                curedge = SampleAnEdge(uni(), uni(), num_edges_up, alias_up, prob_up);
                neg_table = neg_table_up;
                emb_vertex_target = emb_vertex_p;
                emb_context_target = emb_vertex_u;
                edge_source_id = up_edge_source_id;
                edge_target_id = up_edge_target_id;
                break;
        }

        u = edge_source_id[curedge];
        v = edge_target_id[curedge];

        lu = u * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;

        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++) {
            if (d == 0) {
                target = v;
                label = 1;
            } else {
                target = neg_table[Rand(seed)];
                label = 0;
            }
            lv = target * dim;

            Update(&emb_context_target[lu], &emb_vertex_target[lv], vec_error, label,a);

        }
        for (int c = 0; c != dim; c++) emb_context_target[c + lu] += vec_error[c];
        count++;
    }
    free(vec_error);
    return NULL;
}

void OutputFile(char emb_file[100], int num_vertices, ClassVertex *vertex, real *emb_vertex) {
    FILE *fo = fopen(emb_file, "wb");
    std::cout << "outputfile... " << num_vertices << "\n";
    // fprintf(fo, "%d %d\n", num_vertices, dim);
    for (int a = 0; a < num_vertices; a++) {
        //std::cout<<a<<"\n";
        //std::cout<<vertex[a].name<<"\n";
        fprintf(fo, "%s ", vertex[a].name);
        if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
        else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void Output() {
    OutputFile(emb_poi, num_vertices_poi, vertex_pv, emb_vertex_p);
    OutputFile(emb_reg, num_vertices_pr, vertex_pr, emb_vertex_r);
    OutputFile(emb_time, num_vertices_pt, vertex_pt, emb_vertex_t);
    OutputFile(emb_word, num_vertices_pw, vertex_pw, emb_vertex_w);
    OutputFile(emb_user, num_vertices_user, vertex_uv, emb_vertex_u);
}

void TrainLINE() {
    long a;
    // boost::thread *pt = new boost::thread[num_threads];
    std::thread *pt = new std::thread[num_threads];

    printf("--------------------------------\n");
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Initial rho: %lf\n", init_rho);
    printf("Thread: %d\n", num_threads);
    printf("--------------------------------\n");

    InitHashTable();
    ReadPOIs(poi_file);
    max_num_vertices = 1000;
    ReadUsers(user_file);


    ReadData();
    InitAlias();

    InitVector();
    InitNeg();
    InitSigmoidTable();

    clock_t start = clock();
    printf("--------------------------------\n");
    // for (a = 0; a < num_threads; a++) pt[a] = boost::thread(TrainLINEThread, (void *)a);
    for (a = 0; a < num_threads; a++) pt[a] = std::thread(TrainLINEThread, a);
    for (a = 0; a < num_threads; a++) pt[a].join();
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double) (finish - start) / CLOCKS_PER_SEC);

    free(neg_table_pv);
    free(neg_table_pr);
    free(neg_table_pt);
    free(neg_table_pw);

    free(neg_table_uv);
    free(neg_table_ur);
    free(neg_table_ut);
    free(neg_table_uw);
    free(neg_table_up);
    Output();
    std::cout << "GE finish..... " << "\n";
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("LINE: Large Information Network Embedding\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet dimension of vertex embeddings; default is 100\n");
        printf("\t-order <int>\n");
        printf("\t\tNumber of negative examples; default is 5\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-rho <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./GEmodel -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
        return 0;
    }

    if ((i = ArgPos((char *) "-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

    strcpy(poi_file, "POIs.txt");
    strcpy(user_file, "Users.txt");

    strcpy(net_poi, "net_POI.txt");
    strcpy(emb_poi, "net_POI_vec.txt");
    strcpy(net_poi_reg, "net_POI_reg.txt");
    strcpy(emb_reg, "net_reg_vec.txt");
    strcpy(net_poi_time, "net_POI_time.txt");
    strcpy(emb_time, "net_time_vec.txt");
    strcpy(net_poi_word, "net_POI_word.txt");
    strcpy(emb_word, "net_word_vec.txt");

    strcpy(net_user, "net_user.txt");
    strcpy(emb_user, "net_user_vec.txt");
    strcpy(net_user_reg, "net_user_reg.txt");
    strcpy(net_user_time, "net_user_time.txt");
    strcpy(net_user_word, "net_user_word.txt");
    strcpy(net_user_poi, "net_user_poi.txt");


    total_samples *= 1000000;
    rho = init_rho;

    vertex_poi = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_pv = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_pr = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_pt = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_pw = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));

    vertex_user = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_uv = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_ur = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_ut = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_uw = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));
    vertex_up = (struct ClassVertex *) calloc(max_num_vertices, sizeof(struct ClassVertex));

    TrainLINE();
    return 0;
}