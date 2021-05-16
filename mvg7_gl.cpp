#include <GL/glwrapper.h>
#include "rgbd_utils.h"


glm::vec2 clicked_point;
float rot_x =0.0f;
float rot_y =0.0f;
bool drawing_buffer_dirty = true;
glm::vec3 view_translation{ 0,0,-30 };

void cursor_calback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if(!io.WantCaptureMouse) {
        const glm::vec2 p{-xpos, ypos};
        const auto d = clicked_point - p;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
            rot_x += 0.01 * d[1];
            rot_y += 0.01 * d[0];
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
            view_translation[2] += 0.02 * d[1];
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3) == GLFW_PRESS) {
            view_translation[1] += 0.01 * d[1];
            view_translation[0] -= 0.01 * d[0];
        }
        clicked_point = p;
    }
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
int main(int argc, char **argv) {

    Eigen::Matrix<double,6,1> cfg;
    Eigen::Matrix<float,6,1> cfg_imgui;
    cfg_imgui.setZero();
    cfg.setZero();
//#dataset 1
    Eigen::Matrix3d K;
    K << 517.3, 0, 318.6,	0, 516.5, 255.3, 0, 0, 1;
    cv::Mat c2 = cv::imread("rgb/1305031102.175304.png");
    cv::Mat c1 = cv::imread("rgb/1305031102.275326.png");
    cv::Mat d2 = cv::imread("depth/1305031102.160407.png",cv::IMREAD_ANYDEPTH);
    cv::Mat d1 = cv::imread("depth/1305031102.262886.png",cv::IMREAD_ANYDEPTH);
//#dataset 2
//    Eigen::Matrix3d K;
//    K << 535.4,  0, 320.1,0, 539.2, 247.6,0, 0, 1;
//    cv::Mat c2 = cv::imread("rgb/1341847980.722988.png");
//    cv::Mat c1 = cv::imread("rgb/1341847982.998783.png");
//    cv::Mat d2 = cv::imread("depth/1341847980.723020.png",cv::IMREAD_ANYDEPTH);
//    cv::Mat d1 = cv::imread("depth/1341847982.998830.png",cv::IMREAD_ANYDEPTH);


    cv::cvtColor(c1,c1,cv::COLOR_BGR2GRAY);
    cv::cvtColor(c2,c2,cv::COLOR_BGR2GRAY);

    std::vector<rgbd_utils::pyramidLevel> pyr1 {{c1,d1,K}};
    std::vector<rgbd_utils::pyramidLevel> pyr2 {{c2,d2,K}};


    for (int i =1 ;i<5; i++)
    {
        rgbd_utils::pyramidLevel new_lvl_1;
        rgbd_utils::downscale(pyr1[i-1].I,pyr1[i-1].D,pyr1[i-1].K,new_lvl_1.I,new_lvl_1.D,new_lvl_1.K);
        rgbd_utils::pyramidLevel new_lvl_2;
        rgbd_utils::downscale(pyr2[i-1].I,pyr2[i-1].D,pyr2[i-1].K,new_lvl_2.I,new_lvl_2.D,new_lvl_2.K);
        pyr1.emplace_back(new_lvl_1);
        pyr2.emplace_back(new_lvl_2);
    }

    const std::shared_ptr<const cv::Mat> _I1f = std::make_shared<const cv::Mat>(pyr1.front().I);
    const std::shared_ptr<const cv::Mat> _I2f = std::make_shared<const cv::Mat>(pyr2.front().I);
    const std::shared_ptr<const cv::Mat> _D1f = std::make_shared<const cv::Mat>(pyr1.front().D);
    const std::shared_ptr<const cv::Mat> _D2f = std::make_shared<const cv::Mat>(pyr2.front().D);
    const Eigen::Matrix3d _Kf = pyr1.front().K;

    GLFWwindow *window;
    const char *glsl_version = "#version 130";
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(960, 540, "rgbd_demo", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, cursor_calback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) { return -1; }

    GLCall(glClearColor(0.4, 0.4, 0.4, 1));

    Renderer renderer;
    VertexBufferLayout layout;
    layout.Push<float>(3);
    layout.Push<float>(3);

    VertexArray va;
    VertexBuffer vb(gl_primitives::coordinate_system_vertex.data(),
                    gl_primitives::coordinate_system_vertex.size() * sizeof(float));
    va.AddBuffer(vb, layout);
    IndexBuffer ib(gl_primitives::coordinate_system_indices.data(), gl_primitives::coordinate_system_indices.size());

    Shader shader(shader_simple_v, shader_simple_f);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init(glsl_version);

    cv::Mat erromap = rgbd_utils::getErrorImage(_I1f, _I2f, _D1f, _Kf, cfg);
    cv::Mat erromap_jet;
    cv::applyColorMap(erromap, erromap_jet, cv::COLORMAP_JET);
    Texture tex_erromap (erromap_jet);


    bool data_is_dirty = false;

    std::vector<float> draw_buffer_vertices_pc1;
    std::vector<unsigned int> draw_buffer_indices_pc1;

    std::vector<float> draw_buffer_vertices_pc2;
    std::vector<unsigned int> draw_buffer_indices_pc2;

    // prepare data
    const double cx = _Kf(0, 2);
    const double cy = _Kf(1, 2);
    const double fx = _Kf(0, 0);
    const double fy = _Kf(1, 1);
    for (int u = 0; u < _D1f->rows; u++) {
        for (int v = 0; v < _D1f->cols; v++) {
            float i1 = _I1f->at<uint8_t>(int(u), int(v));
            float i2 = _I2f->at<uint8_t>(int(u), int(v));
            float d1 = 1.0f*_D1f->at<uint16_t>(u, v)/rgbd_utils::kDepthScale;
            float d2 = 1.0f*_D2f->at<uint16_t>(u, v)/rgbd_utils::kDepthScale;

            draw_buffer_indices_pc1.push_back(draw_buffer_vertices_pc1.size()/6);
            draw_buffer_indices_pc2.push_back(draw_buffer_vertices_pc2.size()/6);
            Eigen::Matrix<float, 4, 1> p3d_frame1;
            draw_buffer_vertices_pc1.push_back((u - cx) / fx * d1);
            draw_buffer_vertices_pc1.push_back((v - cy) / fy * d1);
            draw_buffer_vertices_pc1.push_back(d1);
            draw_buffer_vertices_pc1.push_back(i1/255);
            draw_buffer_vertices_pc1.push_back(0);
            draw_buffer_vertices_pc1.push_back(i1/255);

            draw_buffer_vertices_pc2.push_back((u - cx) / fx * d2);
            draw_buffer_vertices_pc2.push_back((v - cy) / fy * d2);
            draw_buffer_vertices_pc2.push_back(d2);
            draw_buffer_vertices_pc2.push_back(0);
            draw_buffer_vertices_pc2.push_back(i2/255);
            draw_buffer_vertices_pc2.push_back(i2/255);
        }
    }

    VertexArray va_points1;
    VertexBuffer vb_points1(draw_buffer_vertices_pc1.data(), draw_buffer_vertices_pc1.size() * sizeof(float));
    va_points1.AddBuffer(vb_points1, layout);
    IndexBuffer ib_points1(draw_buffer_indices_pc1.data(), draw_buffer_indices_pc1.size());

    VertexArray va_points2;
    VertexBuffer vb_points2(draw_buffer_vertices_pc2.data(), draw_buffer_vertices_pc2.size() * sizeof(float));
    va_points2.AddBuffer(vb_points2, layout);
    IndexBuffer ib_points2(draw_buffer_indices_pc2.data(), draw_buffer_indices_pc2.size());


    while (!glfwWindowShouldClose(window)) {

        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiIO& io = ImGui::GetIO();
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glm::mat4 proj = glm::perspective(30.f, 1.0f*width/height, 0.05f, 100.0f);
        glm::mat4 model_translate = glm::translate(glm::mat4(1.0f), view_translation);
        glm::mat4 model_rotation_1 = glm::rotate(model_translate, rot_x, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 model_rotation_2 = glm::rotate(model_rotation_1, rot_y, glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 model_rotation_3 = glm::rotate(model_rotation_2, (float)(0.5f*M_PI), glm::vec3(-1.0f, 0.0f, 0.0f));
        glm::mat4 scan2_cfg;
        Eigen::Map<Eigen::Matrix4f>scan2_cfg_map (&scan2_cfg[0][0]);
        scan2_cfg_map = Sophus::SE3d::exp(cfg).matrix().cast<float>();
        shader.Bind(); // bind shader to apply uniform


        // draw reference frame
        GLCall(glPointSize(1));
        shader.setUniformMat4f("u_MVP", proj * model_rotation_2*scan2_cfg);
        renderer.Draw(va_points1, ib_points1, shader, GL_POINTS);
        renderer.Draw(va, ib, shader, GL_LINES);
        shader.setUniformMat4f("u_MVP", proj * model_rotation_2);
        renderer.Draw(va, ib, shader, GL_LINES);
        renderer.Draw(va_points2, ib_points2, shader, GL_POINTS);
        if (data_is_dirty)
        {
            data_is_dirty = false;
            cv::Mat erromap = rgbd_utils::getErrorImage(_I1f, _I2f, _D1f, _Kf, cfg);
            cv::Mat erromap_jet;
            cv::applyColorMap(erromap, erromap_jet, cv::COLORMAP_JET);
            tex_erromap.update(erromap_jet);
        }

        ImGui::Begin("Photometric");
        ImGui::Image((void*)(intptr_t)tex_erromap.getMRendererId(), ImVec2(tex_erromap.GetWidth(), tex_erromap.GetHeight()));
        ImGui::End();
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ImGui::Begin("Calibration Demo");
        ImGui::SliderFloat("param_0", &cfg_imgui.data()[0], -0.4, 0.4);
        ImGui::SliderFloat("param_1", &cfg_imgui.data()[1], -0.4, 0.4);
        ImGui::SliderFloat("param_2", &cfg_imgui.data()[2], -0.4, 0.4);
        ImGui::SliderFloat("param_3", &cfg_imgui.data()[3], -0.4, 0.4);
        ImGui::SliderFloat("param_4", &cfg_imgui.data()[4], -0.4, 0.4);
        ImGui::SliderFloat("param_5", &cfg_imgui.data()[5], -0.4, 0.4);
        if (cfg_imgui != cfg.cast<float>())
        {
            data_is_dirty = true;
            cfg = cfg_imgui.cast<double>();
        }


        if(ImGui::Button("optimize"))
        {
            for (int pyr_level = pyr1.size()-1; pyr_level>=0; pyr_level--)
            {
                std::cout <<"optimizing on level " << pyr_level << std::endl;
                const std::shared_ptr<const cv::Mat> _I1 = std::make_shared<const cv::Mat>(pyr1.at(pyr_level).I);
                const std::shared_ptr<const cv::Mat> _I2 = std::make_shared<const cv::Mat>(pyr2[pyr_level].I);
                const std::shared_ptr<const cv::Mat> _D1 = std::make_shared<const cv::Mat>(pyr1[pyr_level].D);
                const Eigen::Matrix3d _K = pyr1[pyr_level].K;

                ceres::Problem problem;
                problem.AddParameterBlock(cfg.data(), 6, nullptr);

                for (int u = 0; u < _I1->rows; u++) {
                    for (int v = 0; v < _I1->cols; v++) {
                        auto cost_function = rgbd_utils::PhotometricError::Create(_I1, _I2, _D1, _K, u, v);
                        ceres::LossFunction *loss = NULL;//new ceres::HuberLoss(1.0);
                        problem.AddResidualBlock(cost_function, loss, cfg.data());
                    }
                }
                ceres::Solver::Options options;
                options.max_num_iterations = 50;
                options.minimizer_progress_to_stdout = true;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                std::cout << summary.FullReport() << "\n";
                cfg_imgui = cfg.cast<float>();

                cv::Mat i2_p = rgbd_utils::getProjectedImage(_I1f, _I2f, _D1f, _Kf, cfg);
                cv::imwrite("/tmp/i1_p.png", i2_p);
                cv::imwrite("/tmp/i1_r.png", *_I1f);

                data_is_dirty = true;
            }
        }
        if(ImGui::Button("reset"))
        {
            cfg.setZero();
            cfg_imgui.setZero();
            data_is_dirty = true;
        }
        ImGui::End();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;

}