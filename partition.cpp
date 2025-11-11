
/*

Algorithm: Micro-cluster based DBSCAN
Author: Aditya Sarma
email: asaditya1195@gmail.com

*/

#include "partition.h"
#include "vectorc.h"
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <vector>


namespace patch
{
	template < typename T > std::string to_string( const T& n )
	{
		std::ostringstream stm ;
		stm << n ;
		return stm.str() ;
	}
}

int fileReadSingle(char* filename, int* numObjs, int* numCoords, vector< vector<double> >& objects)     /* no. coordinates */
{            
	ifstream in(filename);
	if (!in.is_open())
	{
		fprintf(stderr, "Error: no such file (%s)\n", filename);
		return -1;
	}

	vector<string> lines;
	string line;
	while (std::getline(in, line))
	{
		// trim spaces
		bool allws = true;
		for(char c : line) if (!isspace((unsigned char)c)) { allws = false; break; }
		if (allws) continue;
		if (line.size() > 0 && line[0] == '#') continue; // skip comments
		lines.push_back(line);
	}
	in.close();

	if (lines.empty()) return -1;

	// helper lambdas
	auto is_integer = [](const string &s)->bool{
		if (s.empty()) return false;
		char *endptr = nullptr;
		long v = strtol(s.c_str(), &endptr, 10);
		return endptr != s.c_str() && *endptr == '\0';
	};

	auto split_ws = [](const string &s)->vector<string>{
		vector<string> out;
		string token;
		std::istringstream iss(s);
		while (iss >> token) out.push_back(token);
		return out;
	};

	auto split_csv = [](const string &s)->vector<string>{
		vector<string> out;
		string cur;
		for (char c : s)
		{
			if (c == ',') { if(!cur.empty()) out.push_back(cur); cur.clear(); }
			else cur.push_back(c);
		}
		if(!cur.empty()) out.push_back(cur);
		return out;
	};

	// detect old format: first two non-empty lines are integers
	if (lines.size() >= 2)
	{
		string l0 = lines[0];
		string l1 = lines[1];
		vector<string> t0 = split_ws(l0);
		vector<string> t1 = split_ws(l1);
		if (t0.size() == 1 && t1.size() == 1 && is_integer(t0[0]) && is_integer(t1[0]))
		{
			int totalPoints = atoi(t0[0].c_str());
			int dims = atoi(t1[0].c_str());
			*numObjs = totalPoints;
			*numCoords = dims;
			DIMENSION = dims;

			MINGRIDSIZEglobal = (double*) calloc(DIMENSION, sizeof(double));
			MAXGRIDSIZEglobal = (double*) calloc(DIMENSION, sizeof(double));
			MINGRIDSIZE = (double*) calloc(DIMENSION, sizeof(double));
			MAXGRIDSIZE = (double*) calloc(DIMENSION, sizeof(double));

			objects.resize(totalPoints);
			for (int i = 0; i < totalPoints; i++) objects[i].resize(DIMENSION);

			int idx = 0;
			for (size_t ln = 2; ln < lines.size() && idx < totalPoints; ln++)
			{
				vector<string> vals = split_ws(lines[ln]);
				for (int d = 0; d < DIMENSION && d < (int)vals.size(); d++)
				{
					double v = atof(vals[d].c_str());
					objects[idx][d] = v;
				}

				if (idx == 0)
				{
					for (int d = 0; d < DIMENSION; d++)
					{
						MINGRIDSIZEglobal[d] = objects[0][d];
						MAXGRIDSIZEglobal[d] = objects[0][d];
					}
				}
				for (int d = 0; d < DIMENSION; d++)
				{
					double v = objects[idx][d];
					if (MINGRIDSIZEglobal[d] > v) MINGRIDSIZEglobal[d] = v;
					if (MAXGRIDSIZEglobal[d] < v) MAXGRIDSIZEglobal[d] = v;
				}

				idx++;
			}
			return totalPoints;
		}
	}

	// otherwise treat as CSV: each non-empty line is a point, comma-separated (or whitespace)
	int dims = -1;
	for (size_t ln = 0; ln < lines.size(); ln++)
	{
		string l = lines[ln];
		// try csv split first
		vector<string> toks = split_csv(l);
		if (toks.empty()) toks = split_ws(l);
		if (dims == -1) dims = (int)toks.size();
		if ((int)toks.size() != dims)
		{
			// if inconsistent, try whitespace split
			toks = split_ws(l);
			if ((int)toks.size() != dims)
			{
				// skip malformed line
				continue;
			}
		}
		vector<double> row(dims);
		for (int d = 0; d < dims; d++) row[d] = atof(toks[d].c_str());
		objects.push_back(row);
		if ((int)objects.size() == 1)
		{
			MINGRIDSIZEglobal = (double*) calloc(dims, sizeof(double));
			MAXGRIDSIZEglobal = (double*) calloc(dims, sizeof(double));
			MINGRIDSIZE = (double*) calloc(dims, sizeof(double));
			MAXGRIDSIZE = (double*) calloc(dims, sizeof(double));
			for (int d = 0; d < dims; d++)
			{
				MINGRIDSIZEglobal[d] = row[d];
				MAXGRIDSIZEglobal[d] = row[d];
			}
		}
		else
		{
			for (int d = 0; d < dims; d++)
			{
				if (MINGRIDSIZEglobal[d] > row[d]) MINGRIDSIZEglobal[d] = row[d];
				if (MAXGRIDSIZEglobal[d] < row[d]) MAXGRIDSIZEglobal[d] = row[d];
			}
		}
	}

	*numObjs = (int)objects.size();
	*numCoords = (dims == -1 ? 0 : dims);
	DIMENSION = *numCoords;

	return *numObjs;
}

int fileReadMulti(char* infilename, int* numObjs, int* numCoords, vector< vector<double> >& objects)     /* no. coordinates */
{    
	int i, j, rank, nproc;
	int num_points,  dims;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	/* If the provided path is a regular file, have rank 0 read the whole
	   file and distribute points across ranks. Otherwise keep the
	   original behaviour of opening per-rank files in a directory. */
	struct stat st;
	if (stat(infilename, &st) == 0 && S_ISREG(st.st_mode))
	{
		double t0 = MPI_Wtime();
		int total_points = 0;
		int dims_local = 0;
		vector< vector<double> > all_objects;

	if (rank == 0)
		{
			/* reuse existing single-file reader on rank 0 */
			int rc = fileReadSingle(infilename, &total_points, &dims_local, all_objects);
			if (rc < 0)
			{
				cerr << "rank 0 Error: could not read file: " << infilename << endl;
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
		}

		double t1 = MPI_Wtime();
		if (rank == 0) {
			cerr << "[fileReadMulti] rank 0 read time: " << (t1 - t0) << "s, distributing " << total_points << " points across " << nproc << " ranks\n";
		}

		/* broadcast metadata */
		MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&dims_local, 1, MPI_INT, 0, MPI_COMM_WORLD);

		*numObjs = total_points;
		*numCoords = dims_local;
		DIMENSION = dims_local;

		/* compute balanced counts per rank (in points) */
		vector<int> counts(nproc, 0);
		vector<int> displ(nproc, 0);
		int base = 0, rem = 0;
		if (total_points > 0)
		{
			base = total_points / nproc;
			rem = total_points % nproc;
		}
		for (int r = 0; r < nproc; r++)
		{
			counts[r] = base + (r < rem ? 1 : 0);
			if (r > 0) displ[r] = displ[r-1] + counts[r-1];
		}

		int local_points = counts[rank];

		/* resize local objects */
		objects.resize(local_points);
		for (int ll = 0; ll < local_points; ll++)
			objects[ll].resize(dims_local);

		/* prepare sendcounts/displacements in number of doubles */
		vector<int> sendcounts_dbl(nproc, 0);
		vector<int> displ_dbl(nproc, 0);
		for (int r = 0; r < nproc; r++)
		{
			sendcounts_dbl[r] = counts[r] * dims_local;
			displ_dbl[r] = displ[r] * dims_local;
		}

		vector<double> flat;
		if (rank == 0)
		{
			flat.resize((size_t)total_points * dims_local);
			for (int p = 0; p < total_points; p++)
			{
				for (int d = 0; d < dims_local; d++)
					flat[p * dims_local + d] = all_objects[p][d];
			}
		}

		vector<double> recvbuf;
		recvbuf.resize((size_t)local_points * dims_local);

		MPI_Scatterv(rank == 0 ? &flat[0] : NULL, &sendcounts_dbl[0], &displ_dbl[0], MPI_DOUBLE,
					 recvbuf.data(), (local_points * dims_local), MPI_DOUBLE, 0, MPI_COMM_WORLD);

		/* fill local objects from recvbuf */
		for (int p = 0; p < local_points; p++)
		{
			for (int d = 0; d < dims_local; d++)
				objects[p][d] = recvbuf[p * dims_local + d];
		}

		/* ensure global min/max arrays exist on all ranks — broadcast from root */
		if (rank != 0)
		{
			MINGRIDSIZEglobal = (double*) calloc(dims_local, sizeof(double));
			MAXGRIDSIZEglobal = (double*) calloc(dims_local, sizeof(double));
			MINGRIDSIZE = (double*) calloc(dims_local, sizeof(double));
			MAXGRIDSIZE = (double*) calloc(dims_local, sizeof(double));
		}

		MPI_Bcast(MINGRIDSIZEglobal, dims_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(MAXGRIDSIZEglobal, dims_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		/* copy globals into local per-process arrays if needed */
		for (int d = 0; d < dims_local; d++)
		{
			MINGRIDSIZE[d] = MINGRIDSIZEglobal[d];
			MAXGRIDSIZE[d] = MAXGRIDSIZEglobal[d];
		}

		double t2 = MPI_Wtime();
		if (rank == 0)
			cerr << "[fileReadMulti] rank 0 distribution time (including scatter): " << (t2 - t1) << "s\n";

		return local_points;
	}

	//cout << "Process " << rank << " entered fileReadMulti\n";

	string out("out_");
	string infolder(infilename);
	//cout<<"infolder "<<infolder<<"\n";
	string file_acc_to_rank = out + patch::to_string(rank);
	//cout<<"file_acc_to_rank "<<file_acc_to_rank<<"\n";
	file_acc_to_rank = infolder + "/" + file_acc_to_rank;
	//cout<<"file_acc_to_rank "<<file_acc_to_rank<<"\n";
	int local_points = 0;
	int rc = fileReadSingle((char*)file_acc_to_rank.c_str(), &num_points, &dims, objects);
	if (rc < 0)
	{
		cout << "rank " << rank << " Error: no such file: " << file_acc_to_rank << endl;
		exit(-1);
	}
	local_points = rc;
	*numObjs = num_points;
	*numCoords = dims;
	DIMENSION = dims;
	return local_points;
	return -1;
}

void compute_local_bounding_box(vector< vector<double> >& objects, int* num_points, interval* box)
{
	int i, j;

	for(i = 0; i < DIMENSION; i++)
	{
		box[i].upper = (objects)[0][i];
		box[i].lower = (objects)[0][i];
	}

	for(i = 0; i < DIMENSION; i++)
	{
		for(j = 1; j < *num_points; j++)
		{
			if(box[i].lower > (objects)[j][i])
				box[i].lower = (objects)[j][i];
			else if(box[i].upper < (objects)[j][i])
				box[i].upper = (objects)[j][i];
		}
	}
}

void compute_global_bounding_box(Interval box, Interval gbox, int nproc)
{
	int i, j, k;

	Interval gather_local_box = (Interval) calloc(DIMENSION * nproc, sizeof(struct interval));
	// assert(gather_local_box != NULL);

	MPI_Allgather(box, sizeof(struct interval) * DIMENSION, MPI_BYTE, gather_local_box, 
			sizeof(struct interval) * DIMENSION, MPI_BYTE, MPI_COMM_WORLD);

	for(i = 0; i < DIMENSION; i++)
	{
		gbox[i].lower = gather_local_box[i].lower;
		gbox[i].upper = gather_local_box[i].upper;

		k = i;
		for(j = 0; j < nproc; j++, k += DIMENSION)
		{
			if(gbox[i].lower > gather_local_box[k].lower)
				gbox[i].lower = gather_local_box[k].lower;

			if(gbox[i].upper < gather_local_box[k].upper)
				gbox[i].upper = gather_local_box[k].upper;
		}
	}

	free(gather_local_box);
}

void copy_global_box_to_each_node(Interval* nodes_gbox, Interval gbox, int internal_nodes)
{
	int i, j;
	for(i = 0; i < internal_nodes; i++)
	{
		for(j = 0; j < DIMENSION; j++)
		{
			nodes_gbox[i][j].upper = gbox[j].upper;
			nodes_gbox[i][j].lower = gbox[j].lower;
		}
	}
}

double findKMedian(vectorc* A, int K)
{
	int l,m;
	l=0;

	m = VECTOR_TOTAL(A) - 1;
	while (l<m) 
	{
		double x=A->doubleItems[K];
		int i=l;
		int j=m;
		do {
			while (A->doubleItems[i] < x) i++;
			while (x<A->doubleItems[j]) j--;
			if (i<=j) 
			{
				swap(A->doubleItems[i], A->doubleItems[j]);
				i++; 
				j--;
			}
		} while (i<=j);

		if (j<K) l=i;
		if (K<i) m=j;
	}

	return A->doubleItems[K];
}

double get_median(vector< vector< double> >& objects, int* num_points, int d, MPI_Comm& new_comm)
{	
	double median;

	// vector<double> data;
	vectorc* data = (vectorc*)malloc(sizeof(struct vectorc));
	// assert(data != NULL);

	VECTOR_INIT(data, DOUBLE);
	addHelper[0] = 0; 
	VECTOR_RESIZE(data, *num_points, (void*)addHelper);
	// data.resize(*num_points, 0);

	for (int k=0; k < *num_points; k++)
		data->doubleItems[k] = (objects)[k][d];

	median = findKMedian(data, VECTOR_TOTAL(data)/2);

	VECTOR_FREE(data);
	free(data);

	// data.clear();

	int proc_count;
	MPI_Comm_size(new_comm, &proc_count);

	vectorc* all_medians = (vectorc*)malloc(sizeof(struct vectorc));
	// assert(all_medians!=NULL);

	VECTOR_INIT(all_medians, DOUBLE);
	VECTOR_RESIZE(all_medians, proc_count, (void*)addHelper);

	// vector<double> all_medians;
	// all_medians.resize(proc_count);

	MPI_Allgather(&median, 1, MPI_DOUBLE, &all_medians->doubleItems[0], 1, MPI_DOUBLE, new_comm);	

	median = findKMedian(all_medians, VECTOR_TOTAL(all_medians)/2); 

	VECTOR_FREE(all_medians);
	free(all_medians);

	// all_medians.clear();	

	return median;	
}

int get_points_to_send(vector< vector< double> >& objects, int* num_points, vector<double>& send_buf, vector<int>& invalid_pos_as, double median, int d, int rank, int partner_rank)
{
	int i, count = 0, j;
	send_buf.reserve((*num_points) * DIMENSION); 
	invalid_pos_as.clear();
	invalid_pos_as.resize(*num_points, 0);

	for(i = 0; i < *num_points; i++)
	{
		if (rank < partner_rank)
		{
			if(objects[i][d] > median)
			{
				invalid_pos_as[i] = 1;
				count++;
				for(j = 0; j < DIMENSION; j++)
					send_buf.push_back(objects[i][j]);
			}
		}
		else
		{
			if(objects[i][d] <= median)
			{
				invalid_pos_as[i] = 1;
				count++;
				for(j = 0; j < DIMENSION; j++)
					send_buf.push_back(objects[i][j]);
			}
		}
	}

	return count;
}	

void copy_box(interval* target_box, interval* source_box)
{
	for(int j = 0; j < DIMENSION; j++)
	{
		target_box[j].upper = source_box[j].upper;
		target_box[j].lower = source_box[j].lower;
	}
}

void update_points(vector< vector< double> >& objects, int* num_points, int s_count, vector <int>& invalid_pos_as, vector <double>& recv_buf)
{
	int i, j, k, l, r_count = recv_buf.size() / DIMENSION;

	if(r_count >= s_count)
	{
		invalid_pos_as.resize(*num_points + r_count - s_count, 1);

		objects.resize(*num_points + r_count - s_count);
		for(int ll = 0; ll < *num_points + r_count - s_count; ll++)
			objects[ll].resize(DIMENSION);

		j = 0;
		for(i = 0; i < invalid_pos_as.size(); i++)
		{
			if(invalid_pos_as[i] == 1)
			{
				for(k = 0; k < DIMENSION; k++)
					objects[i][k] = recv_buf[j++];
			}
		}			

		*num_points = *num_points + r_count - s_count;
	}
	else
	{
		j = 0;
		i = 0;	
		if(recv_buf.size() > 0)
		{
			for(i = 0; i < *num_points; i++)
			{
				if(invalid_pos_as[i] == 1)
				{
					for(k = 0; k < DIMENSION; k++)
						objects[i][k] = recv_buf[j++];

					if(j == recv_buf.size())
					{
						i++;
						break;
					}
				}
			}
		}

		l = *num_points;
		for( ; i < invalid_pos_as.size(); i++)
		{
			if(invalid_pos_as[i] == 1)
			{
				while(l > i)
				{
					l--;
					if(invalid_pos_as[l] == 0)
						break;
				}

				if(invalid_pos_as[l] == 0)	
					for(k = 0; k < DIMENSION; k++)
						objects[i][k] = objects[l][k];
			}
		}

		objects.resize(*num_points + r_count - s_count);
		for(int ll = 0; ll < *num_points + r_count - s_count; ll++)
			objects[ll].resize(DIMENSION);

		*num_points = *num_points + r_count - s_count;
	}		
}

void start_partitioning(vector< vector<double> >& objects, int* num_points)
{
	int r_count, s_count, rank, nproc, i, j, k;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	double t_start = MPI_Wtime();

	interval* box = new interval[DIMENSION];

	for(i = 0; i < DIMENSION; i++)
	{	
		box[i].upper = MAXGRIDSIZEglobal[i];
		box[i].lower = MINGRIDSIZEglobal[i];
	}

	interval* gbox = new interval[DIMENSION];
	compute_global_bounding_box(box, gbox, nproc);
	delete [] box;

	int internal_nodes, partner_rank, loops, b, color, sub_rank, d, max, sub_nprocs;
	MPI_Comm new_comm;
	MPI_Status status;

	loops = 0;
	i = nproc;
	internal_nodes = 1;

	while((i = i >> 1) > 0)
	{
		loops++;
		internal_nodes = internal_nodes << 1;
	}

	internal_nodes = internal_nodes << 1;

	interval** nodes_gbox = new interval*[internal_nodes];
	for(i = 0; i < internal_nodes; i++)
		nodes_gbox[i] = new interval[DIMENSION];

	copy_global_box_to_each_node(nodes_gbox, gbox, internal_nodes);
	// now each node in the tree has gbox

	vector <double> send_buf;
	vector <int>   invalid_pos_as;
	vector <double> recv_buf;

	int pow2_i;
	double median;

	for(i = 0; i < loops; i++)
	{
		pow2_i = POW2(i);
		b  = nproc - (int) (nproc / pow2_i);
		color = (int)((rank & b) / POW2(loops - i ));
		partner_rank = rank ^ (int)(nproc/POW2(i + 1));

		MPI_Comm new_comm;
		MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
		MPI_Comm_rank(new_comm, &sub_rank);

		if(sub_rank == 0)
		{
			d = 0;
			for(j = 1; j < DIMENSION; j++)
			{
				if(nodes_gbox[pow2_i + color][j].upper - nodes_gbox[pow2_i + color][j].lower > 
						nodes_gbox[pow2_i + color][d].upper - nodes_gbox[pow2_i + color][d].lower)
					d = j;
			}
		}	

		MPI_Bcast(&d, 1, MPI_INT, 0, new_comm);

		double median  = get_median(objects, num_points, d, new_comm);		

		s_count = get_points_to_send(objects, num_points,send_buf, invalid_pos_as, median, d, rank, partner_rank);

		if (rank < partner_rank)
		{
			MPI_Sendrecv(&s_count, 1, MPI_INT, partner_rank, 4, &r_count, 1, MPI_INT, partner_rank, 5, MPI_COMM_WORLD, &status);
			recv_buf.resize(r_count * DIMENSION, 0.0);
			MPI_Sendrecv(&send_buf[0], s_count * DIMENSION, MPI_DOUBLE, partner_rank, 2,
					&recv_buf[0], r_count * DIMENSION, MPI_DOUBLE, partner_rank, 3, MPI_COMM_WORLD, &status);
			send_buf.clear();
		}
		else
		{
			MPI_Sendrecv(&s_count, 1, MPI_INT, partner_rank, 5, &r_count, 1, MPI_INT, partner_rank, 4, MPI_COMM_WORLD, &status);

			recv_buf.resize(r_count * DIMENSION, 0.0);

			MPI_Sendrecv(&send_buf[0], s_count * DIMENSION, MPI_DOUBLE, partner_rank, 3, 
					&recv_buf[0], r_count * DIMENSION, MPI_DOUBLE, partner_rank, 2, MPI_COMM_WORLD, &status);

			send_buf.clear();
		}

		update_points(objects, num_points, s_count, invalid_pos_as, recv_buf);
		recv_buf.clear();

		copy_box(nodes_gbox[LOWER(pow2_i+color)], nodes_gbox[pow2_i+color]);
		nodes_gbox[LOWER(pow2_i+color)][d].upper =  median;
		copy_box(nodes_gbox[UPPER(pow2_i+color)], nodes_gbox[pow2_i+color]);
		nodes_gbox[UPPER(pow2_i+color)][d].lower =  median;	

		MPI_Comm_free(&new_comm);
	}
	for(i = 0; i < internal_nodes; i++)
		delete [] nodes_gbox[i];

	delete [] nodes_gbox;

	delete [] gbox;

	double t_end = MPI_Wtime();
	if (rank == 0)
		cerr << "[start_partitioning] partitioning time: " << (t_end - t_start) << "s\n";
}

bool addPoints(int source, int buf_size, int dims, vector<double>& raw_data, vector< vector<double> >& remote_objects, vectorc* remote_PrIDs, int* remote_number)
{
	int i, j, k, pos, num_points = buf_size / dims;

	if(DIMENSION != dims)
		return false;

	pos = *remote_number;
	*remote_number = *remote_number + num_points;

	remote_objects.resize(*remote_number);

	for(int ll = 0; ll < *remote_number; ll++)
		remote_objects[ll].resize(dims);

	addHelper[0] = -1;
	VECTOR_RESIZE(remote_PrIDs, *remote_number, (void*)addHelper);
	// remote_PrIDs.resize(*remote_number, -1);

	k = 0;
	for(i = 0; i < num_points; i++)
	{
		for(j = 0; j < dims; j++)
			remote_objects[pos][j] = raw_data[k++];

		remote_PrIDs->intItems[pos] = source;

		pos++;
	}

	return true;
}

bool updatePoints(vector< vector<int> >& raw_ind, vectorc* remote_Indices, vectorc* remote_PrIDs, int remote_number)
{
	int i, source = -1, j = -1, prev_source = -1;

	addHelper[0] = -1;
	VECTOR_RESIZE(remote_Indices, remote_number, (void*)addHelper);

	// remote_Indices.resize(remote_number, -1);

	for(i = 0; i < remote_number; i++)
	{
		source = remote_PrIDs->intItems[i];

		if(source != prev_source)
			j = 0;

		remote_Indices->intItems[i] = raw_ind[source][j++];

		prev_source = source;
	}
	return true;
}

void get_extra_points(vector< vector<double> >& objects, int* num_points, vector< vector<double> >& remote_objects, vectorc* remote_PrIDs, vectorc* remote_Indices, int* remote_number)
{

	int rank, nproc, k, i, j;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	double start = MPI_Wtime();

#ifdef _DEBUG
	if(rank == proc_of_interest) cout << "extra point time part 0 strating " << endl;
#endif

	interval* local_box = new interval[DIMENSION];
	compute_local_bounding_box(objects, num_points, local_box);
	double eps = EPS;

	for(i = 0; i < DIMENSION; i++)
	{
		local_box[i].upper += eps;
		local_box[i].lower -= eps; 
	}

	interval* gather_local_box = new interval[DIMENSION * nproc];

	MPI_Allgather(local_box, sizeof(interval) * DIMENSION, MPI_BYTE, gather_local_box,
			sizeof(interval) * DIMENSION, MPI_BYTE, MPI_COMM_WORLD);

	bool if_inside, overlap;
	int count = 0, gcount;

	vector <double> empty;
	vector <vector <double> > send_buf;
	vector <vector <double> > recv_buf;
	send_buf.resize(nproc, empty);
	recv_buf.resize(nproc, empty);

	vector <int> empty_i;
	vector <vector <int> > send_buf_ind;
	vector <vector <int> > recv_buf_ind;
	send_buf_ind.resize(nproc, empty_i);
	recv_buf_ind.resize(nproc, empty_i);

	double mid1 = MPI_Wtime();
	if (rank == 0)
		cerr << "[get_extra_points] phase1 (gather boxes) time: " << (mid1 - start) << "s\n";

	for(k = 0; k < nproc; k++)
	{
		if (k == rank) // self
			continue;

		overlap = true;
		for(j = 0; j < DIMENSION; j++)
		{
			if(gather_local_box[rank * DIMENSION + j].lower < gather_local_box[k * DIMENSION +j].lower)
			{
				if(gather_local_box[rank * DIMENSION + j].upper - gather_local_box[k * DIMENSION + j].lower < eps)	
				{
					overlap = false;
					break;
				}
			}
			else
			{
				if(gather_local_box[k * DIMENSION + j].upper - gather_local_box[rank * DIMENSION + j].lower < eps)
				{
					overlap = false;
					break;
				}
			}
		}

		if(overlap == false)
			continue;

		// printf("EXTRA POINTS \t %d\n", send_buf[0].size());
		for(i = 0; i < *num_points; i++)
		{
			if_inside = true;
			for(j = 0; j < DIMENSION; j++)
			{
				if(objects[i][j] < gather_local_box[k * DIMENSION + j].lower || 
						objects[i][j] > gather_local_box[k * DIMENSION + j].upper)
				{
					if_inside = false;
					break;
				}
			}

			if(if_inside == true)
			{

				for(j = 0; j < DIMENSION; j++)
				{
					// printf("%lf ", objects[i][j]);
					send_buf[k].push_back(objects[i][j]);
				}

				// printf("\t%d\n", rank);
				send_buf_ind[k].push_back(i);
				count++;
			}
		}
	}

	double mid2 = MPI_Wtime();
	if (rank == 0)
		cerr << "[get_extra_points] phase2 (collect local candidates) time: " << (mid2 - mid1) << "s\n";

	vector <int> send_buf_size, recv_buf_size;
	send_buf_size.resize(nproc, 0);
	recv_buf_size.resize(nproc, 0);

	for(i = 0; i < nproc; i++)
		send_buf_size[i] = send_buf[i].size();

	MPI_Alltoall(&send_buf_size[0], 1, MPI_INT, &recv_buf_size[0], 1, MPI_INT, MPI_COMM_WORLD);

	int tag = 200, send_count, recv_count;
	MPI_Request req_send[2 * nproc], req_recv[2 * nproc];
	MPI_Status stat_send[2 * nproc], stat_recv;

	recv_count = 0;
	for(i = 0; i < nproc; i++)
	{
		if(recv_buf_size[i] > 0)
		{
			recv_buf[i].resize(recv_buf_size[i], 0);
			recv_buf_ind[i].resize(recv_buf_size[i] / DIMENSION, -1);

			MPI_Irecv(&recv_buf[i][0], recv_buf_size[i], MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &req_recv[recv_count++]);
			MPI_Irecv(&recv_buf_ind[i][0], recv_buf_size[i] / DIMENSION, MPI_INT, i, tag + 1, MPI_COMM_WORLD, &req_recv[recv_count++]);
		}
	}

	send_count = 0;

	for(i = 0; i < nproc; i++)
	{
		if(send_buf_size[i] > 0)
		{
			MPI_Isend(&send_buf[i][0], send_buf_size[i], MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &req_send[send_count++]);
			MPI_Isend(&send_buf_ind[i][0], send_buf_size[i] / DIMENSION, MPI_INT, i, tag + 1, MPI_COMM_WORLD, &req_send[send_count++]);
		}
	}

	int rtag, rsource, rpos;


	for(i = 0; i < recv_count; i++)
	{
		MPI_Waitany(recv_count, &req_recv[0], &rpos, &stat_recv);

		rtag = stat_recv.MPI_TAG;
		rsource = stat_recv.MPI_SOURCE;

		if(rtag == tag)
		{
			// #if _GET_EXTRA_POINT_STAT == 0  // WHY THIS IS HERE??????????????????????????????????????????????????????????????
			addPoints(rsource, recv_buf_size[rsource], DIMENSION, recv_buf[rsource], remote_objects, remote_PrIDs, remote_number);
			// #endif
			recv_buf[rsource].clear();
		}
		else if(rtag == tag + 1)
		{
			// postpond this computation and call update points later
			// processing immediately might lead to invalid computation
		}
	}	

	if(send_count > 0)
		MPI_Waitall(send_count, &req_send[0], &stat_send[0]);

	double mid3 = MPI_Wtime();
	if (rank == 0)
		cerr << "[get_extra_points] phase3 (comm) time: " << (mid3 - mid2) << "s\n";

	updatePoints(recv_buf_ind, remote_Indices, remote_PrIDs, *remote_number);

	MPI_Reduce(&count, &gcount, 1, MPI_INT, MPI_SUM, proc_of_interest, MPI_COMM_WORLD);

#ifdef _DEBUG
	if(rank == proc_of_interest)
	{
		cout << "Total extra point " << gcount << endl;
		cout << "Extra point per processor " << gcount/nproc << endl;
	}
#endif

	double end = MPI_Wtime();
	if (rank == 0)
		cerr << "[get_extra_points] total extra points time: " << (end - start) << "s\n";


	empty.clear();
	send_buf.clear();
	recv_buf.clear();
	send_buf_size.clear();
	recv_buf_size.clear();
	send_buf_ind.clear();
	recv_buf_ind.clear();

	delete [] gather_local_box;
	delete [] local_box;
}
