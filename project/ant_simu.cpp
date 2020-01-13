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

using namespace std;

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

    // parallel OMP static default:
    # ifdef _OMP_static_
    #pragma omp parallel for schedule(static) reduction(+:cpteur)
    for ( size_t i = 0; i < ants.size(); ++i )
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    # endif

    // parallel OMP dynamic default:
    # ifdef _OMP_dynamic_
    #pragma omp parallel for schedule(dynamic) reduction(+:cpteur)
    for ( size_t i = 0; i < ants.size(); ++i )
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    # endif

    // parallel OMP with a static schedule and defined step size:
    # ifdef _OMP_static_with_step_
    unsigned number_of_ants = ants.size();
    unsigned number_of_threads;
    # pragma omp parallel shared(number_of_threads)
    {
        number_of_threads = omp_get_num_threads();
        # pragma omp for schedule(static, number_of_ants / number_of_threads) reduction(+:cpteur)
        for ( size_t i = 0; i < number_of_ants; ++i )
            ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    }
    # endif

    // parallel OMP with a dynamic schedule and defined step size:
    # ifdef _OMP_dynamic_with_step_
    unsigned number_of_ants = ants.size();
    unsigned number_of_threads;
    # pragma omp parallel shared(number_of_threads)
    {
        number_of_threads = omp_get_num_threads();
        # pragma omp for schedule(dynamic, number_of_ants / number_of_threads) reduction(+:cpteur)
        for ( size_t i = 0; i < number_of_ants; ++i )
            ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    }
    # endif
    
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
    // chronometre:
    chrono::time_point<std::chrono::system_clock> start, end, start_general, end_general;
    chrono::duration<double> elapsed_seconds;

    const int nb_ants = 2000; // Nombre de fourmis
    const double eps = 0.8;  // Coefficient d'exploration
    const double alpha=0.7; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation
    // Location du nid
    position_t pos_nest{128,128};
    //const int i_nest = 128, j_nest = 128;
    // Location de la nourriture
    position_t pos_food{240,240};
    //const int i_food = 240, j_food = 240;    
    // Génération du territoire 256 x 256 ( 2*(2^7) par direction )
    fractal_land land(7,2,1.,512);
    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
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

    gui::context graphic_context(nargs, argv);
    gui::window& win =  graphic_context.new_window(2*land.dimensions()+10,land.dimensions()+266);
    display_t displayer( land, phen, pos_nest, pos_food, ants, win );
    // Compteur de la quantité de nourriture apportée au nid par les fourmis
    size_t food_quantity = 0;

    gui::event_manager manager;
    // start general clock: 
    start_general = chrono::system_clock::now();

    // exit window:
    manager.on_key_event(int('q'), [] (int code) { exit(0); });
    
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
        advance_time(land, phen, pos_nest, pos_food, ants, food_quantity);
        
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

        // the end condition:
        if (food_quantity >= 10) {
            // end general clock:
            end_general = chrono::system_clock::now();

            // count the difference:
            elapsed_seconds = end_general - start_general;
            cout << "General time since the beginning: " << elapsed_seconds.count() << endl;

            exit(0);
        }

        win.blit(); 
    });
    manager.loop();

    return 0;
}