#include <vector>
#include <iostream>
#include <random>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheromone.hpp"
# include "gui/context.hpp"
# include "gui/colors.hpp"
# include "gui/point.hpp"
# include "gui/segment.hpp"
# include "gui/triangle.hpp"
# include "gui/quad.hpp"
# include "gui/event_manager.hpp"
# include "display.hpp"
#include <chrono>
#include "mpi.h"


# define _OMP_static_
// # define _clock_advance_
// # define _clock_display_

void advance_time( const fractal_land& land, pheromone& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur )
{
    // chronometre:
    chrono::time_point<std::chrono::system_clock> start, end;
    chrono::duration<double> elapsed_seconds;

    // start clock:
    start = chrono::system_clock::now();

    for ( size_t i = 0; i < ants.size(); ++i )
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);

    // end clock:
    end = chrono::system_clock::now();

    // count the difference:
    # ifdef _clock_advance_
    elapsed_seconds = end - start;
    cout << "Advance time: " << elapsed_seconds.count() << endl;
    # endif

    phen.do_evaporation();
    phen.update();
}

int main(int nargs, char* argv[]) {
    const int nb_ants = 2000; // Nombre de fourmis

    const double alpha=0.7; // Coefficient de chaos
    const double beta=0.999; // Coefficient d'évaporation
    // Location du nid
    position_t pos_nest{128,128};
    // const int i_nest = 128, j_nest = 128;
    // Location de la nourriture
    position_t pos_food{240,240};
    // const int i_food = 240, j_food = 240;    
    // Génération du territoire 256 x 256 ( 2*(2^7) par direction )
    fractal_land land(7,2,1.,512);

    // Create two different groups and the global one:
    MPI_Group world_group, group_display_advance, group_advance;
    MPI_Comm comm_display_advance, comm_advance;

    MPI_Init(&nargs, &argv);
    
    int rank, number_of_process;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);

    int ranks_display_advance[2] = {0, 1};
    int ranks_advance[number_of_process - 1];
    int nb_ants_process = nb_ants / (number_of_process - 1);
    const int buffer_size = 1 + 2 * nb_ants + 2 * land.dimensions() * land.dimensions();

    for (unsigned i = 1; i < number_of_process; i++)
        ranks_advance[i - 1] = i;

    // Initialize groups with their ranks:
    if (rank == 1 || rank == 0) {
        MPI_Group_incl(world_group, 2, ranks_display_advance, &group_display_advance);
        MPI_Comm_create_group(MPI_COMM_WORLD, group_display_advance, 0, &comm_display_advance);
    }

    if (rank >= 1) {
        MPI_Group_incl(world_group, number_of_process - 1, ranks_advance, &group_advance);
        MPI_Comm_create_group(MPI_COMM_WORLD, group_advance, 0, &comm_advance);   
    }

    if (rank == 0) {
        // On va créer des fourmis un peu partout sur la carte :
        std::vector<ant> ants;
        ants.reserve(nb_ants);

        // Compteur de la quantité de nourriture apportée au nid par les fourmis
        size_t food_quantity = 0;

        // On crée toutes les fourmis dans la fourmilière.
        pheromone phen(land.dimensions(), pos_food, pos_nest, alpha, beta);
        
        // message buffer:
        vector<double> buffer(buffer_size); 

        MPI_Status status;
        MPI_Recv(buffer.data(), buffer.size(), MPI_DOUBLE, 1, MPI_ANY_TAG, comm_display_advance, &status);

        food_quantity = buffer[0];
        for (unsigned i = 1; i <= nb_ants * 2; i+=2) {
            ants.emplace_back(position_t(buffer[i], buffer[i+1]));
        }

        phen.update_map(vector<double> (buffer.begin() + 1 + 2 * ants.size(), buffer.end()));

        // chronometre:
        chrono::time_point<std::chrono::system_clock> start, end, start_general, end_general;
        chrono::duration<double> elapsed_seconds;

        double max_val = 0.0;
        double min_val = 0.0;
        for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
            for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
                max_val = std::max(max_val, land(i,j));
                min_val = std::min(min_val, land(i,j));
            }
        double delta = max_val - min_val;
        /* On redimensionne les valeurs de fractal_land de sorte que les valeurs
        soient comprises entre zéro et un */
        for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
            for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
                land(i,j) = (land(i,j)-min_val)/delta;
            }

        gui::context graphic_context(nargs, argv);
        gui::window& win =  graphic_context.new_window(2*land.dimensions()+10,land.dimensions()+266);
        display_t displayer( land, phen, pos_nest, pos_food, ants, win );

        gui::event_manager manager;
        // start general clock: 
        start_general = chrono::system_clock::now();

        manager.on_key_event(int('q'), [] (int code) { 
            MPI_Abort(MPI_COMM_WORLD, MPI_SUCCESS);
            exit(0);
        }); // key pressed handler

        // general timer:
        manager.on_key_event(int('t'), [&] (int code) { 
            // end general clock:
            end_general = chrono::system_clock::now();

            // count the difference:
            elapsed_seconds = end_general - start_general;
            cout << "General time since the beginning: " << elapsed_seconds.count() << endl;
        });

        // initialisation of the land:
        manager.on_display([&] { displayer.display(food_quantity); win.blit(); });

        // when we want to advance:
        manager.on_idle([&] () {
            // start clock:
            start = chrono::system_clock::now();
            displayer.display(food_quantity);        
            // end clock:
            end = chrono::system_clock::now();

            // count the difference:
            # ifdef _clock_display_
            elapsed_seconds = end - start;
            cout << "Display time: " << elapsed_seconds.count() << endl;
            # endif

            if (food_quantity >= 10) {
                // end general clock:
                end_general = chrono::system_clock::now();

                // count the difference:
                elapsed_seconds = end_general - start_general;
                cout << "General time since the beginning: " << elapsed_seconds.count() << endl;

                MPI_Abort(MPI_COMM_WORLD, MPI_SUCCESS);
                exit(0);
            }

            win.blit();

            MPI_Recv(buffer.data(), buffer.size(), MPI_DOUBLE, 1, MPI_ANY_TAG, comm_display_advance, &status);

            food_quantity = buffer[0];
            for (unsigned i = 1, j = 0; i < ants.size() * 2; i+=2, ++j) {
                ants[j].set_position(buffer[i], buffer[i + 1]);
            }

            phen.update_map(vector<double> (buffer.begin() + 1 + 2 * ants.size(), buffer.end()));
        }); 

        // loop this algorithm:
        manager.loop();
    } else if (rank != 0) {
        MPI_Request request;
        MPI_Status status;

        const double eps = 0.8;  // Coefficient d'exploration
        // Compteur de la quantité de nourriture apportée au nid par les fourmis
        size_t food_quantity = 0;

        // Définition du coefficient d'exploration de toutes les fourmis.
        ant::set_exploration_coef(eps);

        // On va créer des fourmis un peu partout sur la carte :
        std::vector<ant> ants;

        // For each process we have its own number of ants:
        ants.reserve(nb_ants_process);
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(20);  // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<size_t> ant_pos(0, land.dimensions() - 1);
        for (size_t i = 0; i < nb_ants_process; ++i)
            ants.push_back({{ant_pos(gen),ant_pos(gen)}});

        // On crée toutes les fourmis dans la fourmilière.
        pheromone phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

        // message buffers:
        vector<double> buffer, ants_buffer, ants_buffer_recv, pher_buffer_recv;
        int food_quantity_buffer;

        for ( ; ; ) {
            ants_buffer.clear();
            ants_buffer_recv.clear();
            pher_buffer_recv.clear();

            for (unsigned i = 0; i < nb_ants_process; i++) {
                ants_buffer.emplace_back(ants[i].get_position().first);
                ants_buffer.emplace_back(ants[i].get_position().second);
            }

            if (rank == 1)
                vector<double> (2 * nb_ants).swap(ants_buffer_recv);

            vector<double>(2 * land.dimensions() * land.dimensions()).swap(pher_buffer_recv);

            MPI_Reduce(&food_quantity, &food_quantity_buffer, 1, MPI_INT, MPI_SUM, 0, comm_advance);
            MPI_Gather(ants_buffer.data(), ants_buffer.size(), MPI_DOUBLE, ants_buffer_recv.data(), ants_buffer.size(), MPI_DOUBLE, 0, comm_advance);
            MPI_Allreduce(phen.get_m_map_of_pheromone().data(), pher_buffer_recv.data(), 2 * land.dimensions() * land.dimensions(), MPI_DOUBLE, MPI_MAX, comm_advance);
            phen.update_map(vector<double> (pher_buffer_recv));

            if (rank == 1) {
                buffer.clear();
                buffer.emplace_back((double) food_quantity_buffer);
                buffer.insert(buffer.end(), ants_buffer_recv.begin(), ants_buffer_recv.end());
                buffer.insert(buffer.end(), pher_buffer_recv.begin(), pher_buffer_recv.end());

                MPI_Isend(buffer.data(), buffer.size(), MPI_DOUBLE, 0, 101, comm_display_advance, &request);        
                advance_time(land, phen, pos_nest, pos_food, ants, food_quantity);
                MPI_Wait(&request, &status);
            } else {
                advance_time(land, phen, pos_nest, pos_food, ants, food_quantity);
            }
        }
    }

    MPI_Finalize();
    return 0;
}