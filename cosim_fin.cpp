#include "chrono/physics/ChLoadBodyMesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChSystemNSC.h"

#include "chrono/fea/ChElementShellANCF.h"
#include "chrono/fea/ChLoadContactSurfaceMesh.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono_irrlicht/ChIrrApp.h"

#include "chrono_multicore/physics/ChSystemMulticore.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono_pardisomkl/ChSolverPardisoMKL.h"

using namespace chrono;
using namespace chrono::fea;
using namespace chrono::collision;
using namespace irr;
using namespace chrono::irrlicht; 

int main(int argc, char* argv[]) {
    SetChronoDataPath(CHRONO_DATA_DIR);

    double tankLength = 1.0;
    double tankHeight = 0.5;
    double tankWidth = 2.0;
    double wallThickness = 0.01;
    double grainR = 0.02;

    double finLength = 0.4;
    double finHeight = 0.2;
    double finThickness = 0.01;
    double rho = 1500;
    double E = 5e7;
    double nu = 0.3;

    double tStep = 0.001;
    int maxIter = 30;
    double tol = 1e-3;
    double tFinal = 5.0;

    // FEA system
    ChSystemNSC systemF;

    systemF.Set_G_acc(ChVector<>(0, -9.81, 0));

    auto surfMat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    surfMat->SetFriction(0.3);
    surfMat->SetRestitution(0.0);

    auto fin = chrono_types::make_shared<fea::ChMesh>();

    int numDivX = 5;
    int numDivY = 5;
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
    for (int j = 0; j < numNodeY; j++) { // Loop order MATTERS!!
        for (int i = 0; i < numNodeX; i++) {
            // Location
            double x = i * dx - finLength / 2;
            double y = j * dy;
            double z = 0;

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

    // Shell material
    auto shellMat = chrono_types::make_shared<ChMaterialShellANCF>(rho, E, nu); // rho, E, nu

    // Add elements to mesh
    for (int i = 0; i < numDivX; i++) {
        for (int j = 0; j < numDivY; j++) {
            // Element CCW node index
            int nodeAIndex = i + j * numNodeX;
            int nodeBIndex = i + j * numNodeX + 1;
            int nodeCIndex = i + (j + 1) * numNodeX + 1;
            int nodeDIndex = i + (j + 1) * numNodeX;
            printf("%d, %d\n", i, j);
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
    std::vector<int> vertIndex; // index, pos, vel, force all share the same order
    std::vector<ChVector<>> vertPos;
    std::vector<ChVector<>> vertVel;
    std::vector<ChVector<int>> triangles;
    std::vector<ChVector<>> vertForce;
    meshLoad->OutputSimpleMesh(vertPos, vertVel, triangles);

    auto visColorRed = chrono_types::make_shared<ChColorAsset>();
    visColorRed->SetColor(ChColor(1.0, 0.0, 0.0));
    auto visColorYellow = chrono_types::make_shared<ChColorAsset>();
    visColorYellow->SetColor(ChColor(1.0, 1.0, 0.0));

    for (int i = 0; i < vertPos.size(); i++) {
        vertIndex.push_back(i);
    }

    int bodyId = 0;
    for (int i = 0; i < triangles.size(); i++) {
        auto triangle = chrono_types::make_shared<ChBody>(chrono_types::make_shared<ChCollisionModelMulticore>());
        triangle->SetIdentifier(bodyId);
        //ChVector<int> triangle = triangles[i];
        //printf("%d, %d, %d\n", triangle.x(), triangle.y(), triangle.z());
        //printf("%f, %f, %f\n", vertPos[triangle.x()].x(), vertPos[triangle.x()].y(), vertPos[triangle.x()].z());
        //printf("%f, %f, %f\n", vertPos[triangle.y()].x(), vertPos[triangle.y()].y(), vertPos[triangle.y()].z());
        //printf("%f, %f, %f\n", vertPos[triangle.z()].x(), vertPos[triangle.z()].y(), vertPos[triangle.z()].z());
        int vertAIdx = triangles[i].x();
        int vertBIdx = triangles[i].y();
        int vertCIdx = triangles[i].z();
        ChVector<> pos = (vertPos[vertAIdx] + vertPos[vertBIdx] + vertPos[vertCIdx]) / 3.0;
        ChVector<> vel = (vertVel[vertAIdx] + vertVel[vertBIdx] + vertVel[vertCIdx]) / 3.0;
        triangle->SetPos(pos);
        triangle->SetPos_dt(vel);
        triangle->SetRot(QUNIT);
        triangle->SetCollide(true);
        triangle->SetBodyFixed(true);

        triangle->GetCollisionModel()->ClearModel();
        chrono::utils::AddTriangleGeometry(
            triangle.get(), surfMat,
            vertPos[vertAIdx] - pos, vertPos[vertBIdx] - pos, vertPos[vertCIdx] - pos,
            "triA" + std::to_string(bodyId)
        );
        chrono::utils::AddTriangleGeometry(
            triangle.get(), surfMat,
            vertPos[vertAIdx] - pos, vertPos[vertCIdx] - pos, vertPos[vertBIdx] - pos,
            "triB" + std::to_string(bodyId)
        ); // Same triangle but facing opposite direction so contactable from both directions
        triangle->GetCollisionModel()->BuildModel();

        triangle->AddAsset(visColorRed);
        systemG.AddBody(triangle);
        bodyId++;
    }

    // Add the terrain
    chrono::utils::CreateBoxContainer(
        &systemG, -1, surfMat, 
        ChVector<>(tankLength / 2, tankWidth / 2, tankHeight / 2 - wallThickness / 2), wallThickness, ChVector<>(0, -wallThickness, 0), 
        QUNIT, 
        true, true, true, false
    );
    std::shared_ptr<ChBody> containerBody = systemG.SearchBodyID(-1);
    containerBody->AddAsset(visColorYellow);

    chrono::utils::PDSampler<double> sampler(2 * grainR);
    chrono::utils::Generator generator(&systemG);
    auto m1 = generator.AddMixtureIngredient(chrono::utils::MixtureType::SPHERE, 1.0);
    m1->setDefaultMaterial(surfMat);
    m1->setDefaultDensity(1500);
    m1->setDefaultSize(ChVector<>(grainR, grainR, grainR));

    generator.setBodyIdentifier(bodyId);
    ChVector<> hdims(tankLength / 2 - grainR *1.01, tankHeight / 4, tankWidth / 4 - grainR * 1.01); // Leave a gap from side walls, half filled container
    ChVector<> center(0, tankHeight + tankHeight / 2, tankWidth / 4);
    generator.CreateObjectsBox(sampler, center, hdims);

    //return 0;

    ChSystem* sys = &systemG; // System to visualize
    ChIrrApp application(sys, L"Cosim", core::dimension2d<u32>(1280, 720));
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalCamera(core::vector3dfCH(ChVector<>(tankLength / 1.5, tankHeight * 3, tankWidth / 1.5)), core::vector3dfCH(ChVector<>(0, 0, 0)));
    application.AddShadowAll();
    application.AssetBindAll();
    application.AssetUpdateAll();

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

        real3 force(0, 0, 0);
        systemG.CalculateContactForces();

        vertForce.clear();
        for (int i = 0; i < vertPos.size(); i++) {
            vertForce.push_back(ChVector<>(0, 0, 0));
        }

        for (int i = 0; i < triangles.size(); i++) {
            force = systemG.GetBodyContactForce(i);
            //printf("%f, %f, %f\n", force.x, force.y, force.z);

            vertForce[triangles[i].x()] += ChVector<>(force.x, force.y, force.z) / 3;
            vertForce[triangles[i].y()] += ChVector<>(force.x, force.y, force.z) / 3;
            vertForce[triangles[i].z()] += ChVector<>(force.x, force.y, force.z) / 3;
        }

        // NOTE: no way to input torque
        meshLoad->InputSimpleForces(vertForce, vertIndex);

        systemF.DoStepDynamics(tStep);

        vertPos.clear();
        vertVel.clear();
        triangles.clear();
        meshLoad->OutputSimpleMesh(vertPos, vertVel, triangles);

        for (int i = 0; i < triangles.size(); i++) {
            std::shared_ptr<ChBody> triBody = systemG.Get_bodylist().at(i);

            int vertAIdx = triangles[i].x();
            int vertBIdx = triangles[i].y();
            int vertCIdx = triangles[i].z();
            ChVector<> pos = (vertPos[vertAIdx] + vertPos[vertBIdx] + vertPos[vertCIdx]) / 3.0;
            ChVector<> vel = (vertVel[vertAIdx] + vertVel[vertBIdx] + vertVel[vertCIdx]) / 3.0;
            triBody->SetPos(pos);
            triBody->SetPos_dt(vel);

            // Update visual asset
            for (int j = 0; j < triBody->GetAssets().size(); j++) {
                std::shared_ptr<ChAsset> asset = triBody->GetAssets()[j];
                if (std::dynamic_pointer_cast<ChTriangleMeshShape>(asset)) {
                    std::shared_ptr<geometry::ChTriangleMeshConnected> triMesh = ((ChTriangleMeshShape*)(asset.get()))->GetMesh();
                    triMesh->Clear();
                    triMesh->addTriangle(vertPos[vertAIdx] - pos, vertPos[vertBIdx] - pos, vertPos[vertCIdx] - pos);
                }
                application.AssetUpdate(triBody);
            }

            // Update collision information
            systemG.data_manager->shape_data.triangle_rigid[6 * i + 0] = real3(
                vertPos[vertAIdx].x() - pos.x(),
                vertPos[vertAIdx].y() - pos.y(),
                vertPos[vertAIdx].z() - pos.z()
            );
            systemG.data_manager->shape_data.triangle_rigid[6 * i + 1] = real3(
                vertPos[vertBIdx].x() - pos.x(),
                vertPos[vertBIdx].y() - pos.y(),
                vertPos[vertBIdx].z() - pos.z()
            );
            systemG.data_manager->shape_data.triangle_rigid[6 * i + 2] = real3(
                vertPos[vertCIdx].x() - pos.x(),
                vertPos[vertCIdx].y() - pos.y(),
                vertPos[vertCIdx].z() - pos.z()
            );
            systemG.data_manager->shape_data.triangle_rigid[6 * i + 3] = real3(
                vertPos[vertAIdx].x() - pos.x(),
                vertPos[vertAIdx].y() - pos.y(),
                vertPos[vertAIdx].z() - pos.z()
            );
            systemG.data_manager->shape_data.triangle_rigid[6 * i + 4] = real3(
                vertPos[vertCIdx].x() - pos.x(),
                vertPos[vertCIdx].y() - pos.y(),
                vertPos[vertCIdx].z() - pos.z()
            );
            systemG.data_manager->shape_data.triangle_rigid[6 * i + 5] = real3(
                vertPos[vertBIdx].x() - pos.x(),
                vertPos[vertBIdx].y() - pos.y(),
                vertPos[vertBIdx].z() - pos.z()
            );
        }

        //if (sys->GetChTime() > tFinal) break;
    }

    return 0;
}
