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

    MPI_Init(&nargs, &argv);
    
    int rank, process_number;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_number);

    if (rank == 0) {
        // On va créer des fourmis un peu partout sur la carte :
        std::vector<ant> ants;
        ants.reserve(nb_ants);

        // Compteur de la quantité de nourriture apportée au nid par les fourmis
        size_t food_quantity = 0;

        // On crée toutes les fourmis dans la fourmilière.
        pheromone phen(land.dimensions(), pos_food, pos_nest, alpha, beta);
        
        // message buffer:
        vector<double> buffer(1 + 2 * nb_ants + 2 * land.dimensions() * land.dimensions()); 

        MPI_Status status;
        MPI_Recv(buffer.data(), buffer.size(), MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

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

            if (food_quantity >= 1000) {
                // end general clock:
                end_general = chrono::system_clock::now();

                // count the difference:
                elapsed_seconds = end_general - start_general;
                cout << "General time since the beginning: " << elapsed_seconds.count() << endl;

                MPI_Abort(MPI_COMM_WORLD, MPI_SUCCESS);
                exit(0);
            }

            win.blit();

            MPI_Recv(buffer.data(), buffer.size(), MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            food_quantity = buffer[0];
            for (unsigned i = 1, j = 0; i < ants.size() * 2; i+=2, ++j) {
                ants[j].set_position(buffer[i], buffer[i + 1]);
            }

            phen.update_map(vector<double> (buffer.begin() + 1 + 2 * ants.size(), buffer.end()));
        }); 

        // loop this algorithm:
        manager.loop();
    } else if (rank == 1) {
        const double eps = 0.8;  // Coefficient d'exploration
        // Compteur de la quantité de nourriture apportée au nid par les fourmis
        size_t food_quantity = 0;

        // Définition du coefficient d'exploration de toutes les fourmis.
        ant::set_exploration_coef(eps);

        // On va créer des fourmis un peu partout sur la carte :
        std::vector<ant> ants;
        ants.reserve(nb_ants);
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(20);  // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<size_t> ant_pos( 0, land.dimensions()-1 );
        for ( size_t i = 0; i < nb_ants; ++i )
            ants.push_back({{ant_pos(gen),ant_pos(gen)}});

        // On crée toutes les fourmis dans la fourmilière.
        pheromone phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

        // message buffer:
        vector<double> buffer;

        for ( ; ; ) {
            buffer.clear();

            buffer.emplace_back(food_quantity);
            for (unsigned i = 0; i < ants.size(); i++) {
                buffer.emplace_back(ants[i].get_position().first);
                buffer.emplace_back(ants[i].get_position().second);
            }

            for (unsigned i = 0; i < land.dimensions(); i++)
                for (unsigned j = 0; j < land.dimensions(); j++) {
                    buffer.emplace_back((double) phen(i, j)[0]);
                    buffer.emplace_back((double) phen(i, j)[1]);
                }

            MPI_Request request;
            MPI_Status status;
            MPI_Isend(buffer.data(), buffer.size(), MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, &request);        
            advance_time(land, phen, pos_nest, pos_food, ants, food_quantity);
            MPI_Wait(&request, &status);
        }
    }

    MPI_Finalize();
    return 0;
}