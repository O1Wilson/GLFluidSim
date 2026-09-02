#include "Application.h"

int main() {
    Application app(800, 600, "2D Fluid Sim", 128);

    if (!app.init()) {
        return -1;
    }

    app.run();
    return 0;
}
