
#include "chrono/geometry/ChTriangleMeshConnected.h"
#include "chrono/physics/ChLoadBodyMesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChLoaderUV.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono/fea/ChElementTetra_4.h"
#include "chrono/fea/ChLoadContactSurfaceMesh.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChMeshFileLoader.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono_irrlicht/ChIrrApp.h"

#include "chrono_multicore/physics/ChSystemMulticore.h"

#include "chrono/ChConfig.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_pardisomkl/ChSolverPardisoMKL.h"

using namespace chrono;
using namespace chrono::fea;
using namespace chrono::collision;
using namespace irr;
using namespace chrono::irrlicht; 

int main(int argc, char* argv[]) {
    SetChronoDataPath(CHRONO_DATA_DIR);

    double xDim = 1.0;
    double yDim = 0.5;
    double zDim = 2.0;
    double wallT = 0.01;
    double grainR = 0.02;

    double finLength = 0.5;
    double finHeight = 0.5;
    double finThickness = 0.01;
    double rho = 1500;
    double E = 1e9;
    double nu = 0.3;

    double tStep = 0.005;
    int maxIter = 30;
    double tol = 1e-3;
    double tFinal = 5.0;

    // FEA system
    ChSystemSMC systemF;

    systemF.Set_G_acc(ChVector<>(0, -9.81, 0));

    auto surfMat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    surfMat->SetFriction(0.3);
    surfMat->SetRestitution(0.0);

    auto fin = chrono_types::make_shared<fea::ChMesh>();

    int numDivX = 2;
    int numDivY = 2;
    int numDivZ = 0;
    int numNodeX = numDivX + 1;
    int numNodeY = numDivY + 1;
    int numNodeZ = numDivZ + 1;
    int numElements = numDivX * numDivY * (numDivZ + 1);
    int numNodes = numNodeX * numNodeY * numNodeZ;

    double dx = finLength / numDivX;
    double dy = finHeight / numDivY;
    double dz = numDivZ == 0 ? finThickness : finThickness / numDivZ;

    // Add nodes to mesh
    for (int k = 0; k < numNodeZ; k++) { // Loop order MATTERS!!
        for (int j = 0; j < numNodeY; j++) {
            for (int i = 0; i < numNodeX; i++) {
                // Location
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;

                // Direction
                double dirX = 0;
                double dirY = 0;
                double dirZ = 1;
                printf("%f, %f, %f\n", x, y, z);

                // Node
                auto node = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector<>(x, y, z), ChVector<>(dirX, dirY, dirZ));
                node->SetMass(0.0);

                // Constraint one end
                if (i == 0) {
                    node->SetFixed(true);
                }

                fin->AddNode(node);
            }
        }
    }

    // Shell material
    auto shellMat = chrono_types::make_shared<ChMaterialShellANCF>(rho, E, nu); // rho, E, nu

    // Add elements to mesh
    for (int i = 0; i < numDivX; i++) {
        for (int j = 0; j < numDivY; j++) {
            for (int k = 0; k < numDivZ + 1; k++) {
                // Element CCW node index
                int nodeAIndex = i + k * numNodeX * numNodeY + j * numNodeX;
                int nodeBIndex = i + k * numNodeX * numNodeY + j * numNodeX + 1;
                int nodeCIndex = i + k * numNodeX * numNodeY + (j + 1) * numNodeX + 1;
                int nodeDIndex = i + k * numNodeX * numNodeY + (j + 1) * numNodeX;
                printf("%d, %d, %d\n", i, j, k);
                printf("%d, %d, %d, %d\n", nodeAIndex, nodeBIndex, nodeCIndex, nodeDIndex);

                // Element
                auto element = chrono_types::make_shared<ChElementShellANCF>();
                element->SetNodes(
                    std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(nodeAIndex)),
                    std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(nodeBIndex)),
                    std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(nodeCIndex)),
                    std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(nodeDIndex))
                );
                element->SetDimensions(dx, dy);
                element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, shellMat);
                element->SetAlphaDamp(0.01);
                element->SetGravityOn(false);

                fin->AddElement(element);
            }
        }
    }
    systemF.Add(fin);

    // Add contact
    auto contactSurf = chrono_types::make_shared<ChContactSurfaceMesh>(surfMat);
    fin->AddContactSurface(contactSurf);
    contactSurf->AddFacesFromBoundary();

    // Load container to communicate between systems 
    auto loadContainer = chrono_types::make_shared<ChLoadContainer>();
    systemF.Add(loadContainer);
    auto meshLoad = chrono_types::make_shared<ChLoadContactSurfaceMesh>(contactSurf);
    loadContainer->Add(meshLoad);

    // Visualizations
    auto visWireframe = chrono_types::make_shared<ChVisualizationFEAmesh>(*(fin.get()));
    visWireframe->SetWireframe(true);
    fin->AddAsset(visWireframe);

    // Solver
    auto mklSolver = chrono_types::make_shared<ChSolverPardisoMKL>();
    mklSolver->LockSparsityPattern(true);
    systemF.SetSolver(mklSolver);

    // Time stepper
    systemF.SetTimestepperType(ChTimestepper::Type::HHT);
    auto stepper = std::static_pointer_cast<ChTimestepperHHT>(systemF.GetTimestepper());
    stepper->SetStepControl(false);

    // Multi core system
    ChSystemMulticoreNSC systemG;

    // Set gravitational acceleration
    systemG.Set_G_acc(systemF.Get_G_acc());

    // Set solver parameters
    systemG.GetSettings()->solver.solver_mode = SolverMode::SLIDING;
    systemG.GetSettings()->solver.max_iteration_normal = maxIter / 3;
    systemG.GetSettings()->solver.max_iteration_sliding = maxIter / 3;
    systemG.GetSettings()->solver.max_iteration_spinning = 0;
    systemG.GetSettings()->solver.max_iteration_bilateral = maxIter / 3;
    systemG.GetSettings()->solver.tolerance = tol;
    systemG.GetSettings()->solver.alpha = 0;
    systemG.GetSettings()->solver.contact_recovery_speed = 10000;
    systemG.ChangeSolverType(SolverType::APGD);
    systemG.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;

    systemG.GetSettings()->collision.collision_envelope = 0.01;
    systemG.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    // Add fin mesh
    std::vector<ChVector<>> vertPos;
    std::vector<ChVector<>> vertVel;
    std::vector<ChVector<int>> triangles;
    meshLoad->OutputSimpleMesh(vertPos, vertVel, triangles);

    int bodyId = 0;
    for (int i = 0; i < triangles.size(); i++) {
        ChVector<int> triangle = triangles[i];
        printf("%d, %d, %d\n", triangle.x(), triangle.y(), triangle.z());
        printf("%f, %f, %f\n", vertPos[triangle.x()].x(), vertPos[triangle.x()].y(), vertPos[triangle.x()].z());
        printf("%f, %f, %f\n", vertPos[triangle.y()].x(), vertPos[triangle.y()].y(), vertPos[triangle.y()].z());
        printf("%f, %f, %f\n", vertPos[triangle.z()].x(), vertPos[triangle.z()].y(), vertPos[triangle.z()].z());
    }

    // Add the terrain, MUST BE ADDED AFTER TIRE GEOMETRY (for index assumptions)
    chrono::utils::CreateBoxContainer(&systemG, -1, surfMat, ChVector<>(xDim / 2, zDim / 2, yDim /2), wallT, ChVector<>(0, 0, 0), QUNIT, true,
                                      true, true, false);

    chrono::utils::PDSampler<double> sampler(2 * grainR);
    chrono::utils::Generator generator(&systemG);
    auto m1 = generator.AddMixtureIngredient(chrono::utils::MixtureType::SPHERE, 1.0);
    m1->setDefaultMaterial(surfMat);
    m1->setDefaultDensity(1500);
    m1->setDefaultSize(ChVector<>(grainR, grainR * 2, grainR));

    generator.setBodyIdentifier(0);
    ChVector<> hdims(xDim / 2 - grainR *1.01, yDim / 4, zDim / 2 - grainR * 1.01); // Leave a gap from side walls, half filled container
    ChVector<> center(0, yDim + yDim / 2, 0);
    generator.CreateObjectsBox(sampler, center, hdims);

    //return 0;

    ChSystem* sys = &systemG; // System to visualize
    ChIrrApp application(sys, L"Cosim", core::dimension2d<u32>(1280, 720));
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalCamera(core::vector3dfCH(ChVector<>(xDim / 1.5, yDim * 3, zDim / 1.5)), core::vector3dfCH(ChVector<>(0, 0, 0)));
    application.AssetBindAll();
    application.AssetUpdateAll();
    application.AddShadowAll();

    application.SetTimestep(tStep);

    while (application.GetDevice()->run()) {
        printf("time: %f\n", sys->GetChTime());

        application.BeginScene();
        application.DrawAll();

        tools::drawSegment(application.GetVideoDriver(), ChVector<>(0, 0, 0), ChVector<>(1, 0, 0), irr::video::SColor(255, 255, 0, 0));
        tools::drawSegment(application.GetVideoDriver(), ChVector<>(0, 0, 0), ChVector<>(0, 1, 0), irr::video::SColor(255, 0, 255, 0));
        tools::drawSegment(application.GetVideoDriver(), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), irr::video::SColor(255, 0, 0, 255));
        
        application.DoStep();
        application.EndScene();

        if (sys->GetChTime() > tFinal) break;
    }

    return 0;
}
