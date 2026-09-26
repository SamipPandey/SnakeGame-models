#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#endif

using namespace std;


static constexpr int BOARD_W = 10;
static constexpr int BOARD_H = 10;

static constexpr int INPUTS = 25;
static constexpr int ACTIONS = 3;

static constexpr int H1 = 96;
static constexpr int H2 = 96;
static constexpr int H3 = 48;

static constexpr int MAX_EPISODES = 30;

static constexpr int MAX_STEPS_PER_EPISODE = 2500;

static constexpr int REPLAY_CAPACITY = 50000;
static constexpr int BATCH_SIZE = 64;

static constexpr float GAMMA = 0.95f;
static constexpr float LEARNING_RATE = 0.0035f;

static constexpr float EPSILON_START = 1.0f;
static constexpr float EPSILON_END = 0.0003f;
static constexpr int EPSILON_DECAY_STEPS = 250000;

static constexpr int TARGET_UPDATE_EVERY = 1500;

static constexpr int PRINT_EVERY = 1;
static constexpr int EVALUATE_EVERY = 1000;
static constexpr int EVALUATION_GAMES = 50;

static constexpr int VISUAL_DELAY_MS = 80;

// TYPES

struct Point {
    int x;
    int y;

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

enum Direction {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3
};

enum Action {
    STRAIGHT = 0,
    TURN_LEFT = 1,
    TURN_RIGHT = 2
};

struct Transition {
    array<float, INPUTS> state{};
    int action = 0;
    float reward = 0.0f;
    array<float, INPUTS> nextState{};
    bool done = false;
};

// RANDOM


class Random {
private:
    mt19937 rng;

public:
    Random() {
        random_device rd;
        rng.seed(rd());
    }

    int integer(int lo, int hi) {
        uniform_int_distribution<int> d(lo, hi);
        return d(rng);
    }

    float real(float lo, float hi) {
        uniform_real_distribution<float> d(lo, hi);
        return d(rng);
    }

    bool chance(float p) {
        return real(0.0f, 1.0f) < p;
    }

    template<typename T>
    void shuffle(vector<T>& v) {
        std::shuffle(v.begin(), v.end(), rng);
    }
};

static Random RNG;

    
// MATH
    

static float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static float reluDerivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

    
// SIMPLE DENSE LAYER
    

struct Layer {
    int inSize;
    int outSize;

    vector<float> W;
    vector<float> B;

    vector<float> mW;
    vector<float> vW;

    vector<float> mB;
    vector<float> vB;

    Layer() : inSize(0), outSize(0) {}

    Layer(int in, int out) {
        initialize(in, out);
    }

    void initialize(int in, int out) {
        inSize = in;
        outSize = out;

        W.resize(in * out);
        B.assign(out, 0.0f);

        mW.assign(in * out, 0.0f);
        vW.assign(in * out, 0.0f);

        mB.assign(out, 0.0f);
        vB.assign(out, 0.0f);

        float scale = sqrtf(2.0f / static_cast<float>(in));

        for (float& w : W) {
            w = RNG.real(-scale, scale);
        }
    }

    vector<float> forward(const vector<float>& input) const {
        vector<float> output(outSize, 0.0f);

        for (int j = 0; j < outSize; ++j) {
            float sum = B[j];

            for (int i = 0; i < inSize; ++i) {
                sum += input[i] * W[i * outSize + j];
            }

            output[j] = sum;
        }

        return output;
    }
};

    
// DQN
    

class DQN {
private:
    Layer l1;
    Layer l2;
    Layer l3;
    Layer out;

    long long adamStep = 0;

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float adamEps = 1e-8f;

public:
    DQN()
        : l1(INPUTS, H1),
          l2(H1, H2),
          l3(H2, H3),
          out(H3, ACTIONS) {}

    vector<float> predict(const array<float, INPUTS>& input) const {
        vector<float> x(input.begin(), input.end());

        x = l1.forward(x);
        for (float& v : x) v = relu(v);

        x = l2.forward(x);
        for (float& v : x) v = relu(v);

        x = l3.forward(x);
        for (float& v : x) v = relu(v);

        return out.forward(x);
    }

    float qValue(
        const array<float, INPUTS>& state,
        int action
    ) const {
        return predict(state)[action];
    }

    void trainBatch(
        const vector<Transition>& batch,
        const DQN& targetNetwork
    ) {
        if (batch.empty()) return;

        // Accumulated gradients.
        vector<float> gW1(l1.W.size(), 0.0f);
        vector<float> gB1(l1.B.size(), 0.0f);

        vector<float> gW2(l2.W.size(), 0.0f);
        vector<float> gB2(l2.B.size(), 0.0f);

        vector<float> gW3(l3.W.size(), 0.0f);
        vector<float> gB3(l3.B.size(), 0.0f);

        vector<float> gWo(out.W.size(), 0.0f);
        vector<float> gBo(out.B.size(), 0.0f);

        for (const Transition& t : batch) {

            // Forward pass

            vector<float> x(t.state.begin(), t.state.end());

            vector<float> z1 = l1.forward(x);
            vector<float> a1 = z1;

            for (float& v : a1) {
                v = relu(v);
            }

            vector<float> z2 = l2.forward(a1);
            vector<float> a2 = z2;

            for (float& v : a2) {
                v = relu(v);
            }

            vector<float> z3 = l3.forward(a2);
            vector<float> a3 = z3;

            for (float& v : a3) {
                v = relu(v);
            }

            vector<float> q = out.forward(a3);

            // Double DQN target

            float target = t.reward;

            if (!t.done) {
                vector<float> onlineNext = predict(t.nextState);

                int bestNextAction =
                    static_cast<int>(
                        max_element(
                            onlineNext.begin(),
                            onlineNext.end()
                        ) - onlineNext.begin()
                    );

                float targetQ =
                    targetNetwork.qValue(
                        t.nextState,
                        bestNextAction
                    );

                target += GAMMA * targetQ;
            }

            // Huber loss derivative.
            float error = q[t.action] - target;

            float gradQ = 0.0f;

            if (fabs(error) <= 1.0f) {
                gradQ = error;
            } else {
                gradQ = error > 0.0f ? 1.0f : -1.0f;
            }

            // Gradient of output layer.
            vector<float> dzOut(ACTIONS, 0.0f);
            dzOut[t.action] = gradQ;

            for (int i = 0; i < H3; ++i) {
                for (int j = 0; j < ACTIONS; ++j) {
                    gWo[i * ACTIONS + j] +=
                        a3[i] * dzOut[j];
                }
            }

            for (int j = 0; j < ACTIONS; ++j) {
                gBo[j] += dzOut[j];
            }

            // Backprop into layer 3.
            vector<float> da3(H3, 0.0f);

            for (int i = 0; i < H3; ++i) {
                float sum = 0.0f;

                for (int j = 0; j < ACTIONS; ++j) {
                    sum += out.W[i * ACTIONS + j] * dzOut[j];
                }

                da3[i] = sum;
            }

            vector<float> dz3(H3, 0.0f);

            for (int i = 0; i < H3; ++i) {
                dz3[i] =
                    da3[i] * reluDerivative(z3[i]);
            }

            for (int i = 0; i < H2; ++i) {
                for (int j = 0; j < H3; ++j) {
                    gW3[i * H3 + j] +=
                        a2[i] * dz3[j];
                }
            }

            for (int j = 0; j < H3; ++j) {
                gB3[j] += dz3[j];
            }

            // Backprop layer 2.
            vector<float> da2(H2, 0.0f);

            for (int i = 0; i < H2; ++i) {
                float sum = 0.0f;

                for (int j = 0; j < H3; ++j) {
                    sum += l3.W[i * H3 + j] * dz3[j];
                }

                da2[i] = sum;
            }

            vector<float> dz2(H2, 0.0f);

            for (int i = 0; i < H2; ++i) {
                dz2[i] =
                    da2[i] * reluDerivative(z2[i]);
            }

            for (int i = 0; i < H1; ++i) {
                for (int j = 0; j < H2; ++j) {
                    gW2[i * H2 + j] +=
                        a1[i] * dz2[j];
                }
            }

            for (int j = 0; j < H2; ++j) {
                gB2[j] += dz2[j];
            }

            // Backprop layer 1.
            vector<float> da1(H1, 0.0f);

            for (int i = 0; i < H1; ++i) {
                float sum = 0.0f;

                for (int j = 0; j < H2; ++j) {
                    sum += l2.W[i * H2 + j] * dz2[j];
                }

                da1[i] = sum;
            }

            vector<float> dz1(H1, 0.0f);

            for (int i = 0; i < H1; ++i) {
                dz1[i] =
                    da1[i] * reluDerivative(z1[i]);
            }

            for (int i = 0; i < INPUTS; ++i) {
                for (int j = 0; j < H1; ++j) {
                    gW1[i * H1 + j] +=
                        x[i] * dz1[j];
                }
            }

            for (int j = 0; j < H1; ++j) {
                gB1[j] += dz1[j];
            }
        }

        float invBatch =
            1.0f / static_cast<float>(batch.size());

        for (float& v : gW1) v *= invBatch;
        for (float& v : gB1) v *= invBatch;

        for (float& v : gW2) v *= invBatch;
        for (float& v : gB2) v *= invBatch;

        for (float& v : gW3) v *= invBatch;
        for (float& v : gB3) v *= invBatch;

        for (float& v : gWo) v *= invBatch;
        for (float& v : gBo) v *= invBatch;

        // Gradient clipping.
        auto clip = [](vector<float>& g) {
            for (float& v : g) {
                v = max(-5.0f, min(5.0f, v));
            }
        };

        clip(gW1);
        clip(gB1);
        clip(gW2);
        clip(gB2);
        clip(gW3);
        clip(gB3);
        clip(gWo);
        clip(gBo);

        ++adamStep;

        adamUpdate(l1, gW1, gB1);
        adamUpdate(l2, gW2, gB2);
        adamUpdate(l3, gW3, gB3);
        adamUpdate(out, gWo, gBo);
    }

    void adamUpdate(
        Layer& layer,
        const vector<float>& gW,
        const vector<float>& gB
    ) {
        float b1Correction =
            1.0f - powf(beta1, static_cast<float>(adamStep));

        float b2Correction =
            1.0f - powf(beta2, static_cast<float>(adamStep));

        for (size_t i = 0; i < layer.W.size(); ++i) {
            layer.mW[i] =
                beta1 * layer.mW[i] +
                (1.0f - beta1) * gW[i];

            layer.vW[i] =
                beta2 * layer.vW[i] +
                (1.0f - beta2) * gW[i] * gW[i];

            float mHat =
                layer.mW[i] / b1Correction;

            float vHat =
                layer.vW[i] / b2Correction;

            layer.W[i] -=
                LEARNING_RATE *
                mHat /
                (sqrtf(vHat) + adamEps);
        }

        for (size_t i = 0; i < layer.B.size(); ++i) {
            layer.mB[i] =
                beta1 * layer.mB[i] +
                (1.0f - beta1) * gB[i];

            layer.vB[i] =
                beta2 * layer.vB[i] +
                (1.0f - beta2) * gB[i] * gB[i];

            float mHat =
                layer.mB[i] / b1Correction;

            float vHat =
                layer.vB[i] / b2Correction;

            layer.B[i] -=
                LEARNING_RATE *
                mHat /
                (sqrtf(vHat) + adamEps);
        }
    }

    void copyFrom(const DQN& other) {
        l1 = other.l1;
        l2 = other.l2;
        l3 = other.l3;
        out = other.out;
        adamStep = other.adamStep;
    }

    bool save(const string& filename) const {
        ofstream file(filename);

        if (!file) {
            return false;
        }

        file << setprecision(9);

        auto saveLayer =
            [&](const Layer& layer) {

                file << layer.inSize << " "
                     << layer.outSize << "\n";

                for (float v : layer.W) {
                    file << v << " ";
                }

                file << "\n";

                for (float v : layer.B) {
                    file << v << " ";
                }

                file << "\n";
            };

        saveLayer(l1);
        saveLayer(l2);
        saveLayer(l3);
        saveLayer(out);

        return true;
    }

    bool load(const string& filename) {
        ifstream file(filename);

        if (!file) {
            return false;
        }

        auto loadLayer =
            [&](Layer& layer) {

                int in;
                int outSize;

                file >> in >> outSize;

                if (in != layer.inSize ||
                    outSize != layer.outSize) {
                    return false;
                }

                for (float& v : layer.W) {
                    file >> v;
                }

                for (float& v : layer.B) {
                    file >> v;
                }

                return true;
            };

        if (!loadLayer(l1)) return false;
        if (!loadLayer(l2)) return false;
        if (!loadLayer(l3)) return false;
        if (!loadLayer(out)) return false;

        return true;
    }
};

    
// REPLAY BUFFER
    

class ReplayBuffer {
private:
    deque<Transition> buffer;

public:
    void push(const Transition& t) {
        if (buffer.size() >= REPLAY_CAPACITY) {
            buffer.pop_front();
        }

        buffer.push_back(t);
    }

    size_t size() const {
        return buffer.size();
    }

    vector<Transition> sample(int count) {
        vector<Transition> result;

        if (buffer.empty()) {
            return result;
        }

        count =
            min(
                count,
                static_cast<int>(buffer.size())
            );

        result.reserve(count);

        for (int i = 0; i < count; ++i) {
            int index =
                RNG.integer(
                    0,
                    static_cast<int>(buffer.size()) - 1
                );

            result.push_back(buffer[index]);
        }

        return result;
    }
};

    
// SNAKE GAME
    

class SnakeGame {
private:
    deque<Point> snake;

    Point food{};

    Direction direction = RIGHT;

    bool finished = false;

    int score = 0;
    int steps = 0;
    int staleSteps = 0;

public:
    SnakeGame() {
        reset();
    }

    void reset() {
        snake.clear();

        Point center{
            BOARD_W / 2,
            BOARD_H / 2
        };

        snake.push_back(center);
        snake.push_back({center.x - 1, center.y});
        snake.push_back({center.x - 2, center.y});

        direction = RIGHT;

        score = 0;
        steps = 0;
        staleSteps = 0;
        finished = false;

        spawnFood();
    }

    const deque<Point>& getSnake() const {
        return snake;
    }

    Point getFood() const {
        return food;
    }

    Direction getDirection() const {
        return direction;
    }

    int getScore() const {
        return score;
    }

    int getSteps() const {
        return steps;
    }

    bool isFinished() const {
        return finished;
    }

    Point nextPositionForAction(int action) const {
        Direction d = direction;

        if (action == TURN_LEFT) {
            d = static_cast<Direction>(
                (static_cast<int>(direction) + 3) % 4
            );
        }
        else if (action == TURN_RIGHT) {
            d = static_cast<Direction>(
                (static_cast<int>(direction) + 1) % 4
            );
        }

        Point p = snake.front();

        if (d == UP) {
            p.y--;
        }
        else if (d == RIGHT) {
            p.x++;
        }
        else if (d == DOWN) {
            p.y++;
        }
        else {
            p.x--;
        }

        return p;
    }

    Direction directionForAction(int action) const {
        if (action == TURN_LEFT) {
            return static_cast<Direction>(
                (static_cast<int>(direction) + 3) % 4
            );
        }

        if (action == TURN_RIGHT) {
            return static_cast<Direction>(
                (static_cast<int>(direction) + 1) % 4
            );
        }

        return direction;
    }

    bool occupied(
        const Point& p,
        bool includeTail = true
    ) const {
        int limit =
            static_cast<int>(snake.size());

        if (!includeTail && limit > 0) {
            limit--;
        }

        for (int i = 0; i < limit; ++i) {
            if (snake[i] == p) {
                return true;
            }
        }

        return false;
    }

    bool inside(const Point& p) const {
        return
            p.x >= 0 &&
            p.x < BOARD_W &&
            p.y >= 0 &&
            p.y < BOARD_H;
    }

    bool isImmediateSafe(int action) const {
        Point next =
            nextPositionForAction(action);

        bool eating = next == food;

        if (!inside(next)) {
            return false;
        }

        // If not eating the tail moves away and can be entered
        return !occupied(next, eating);
    }

    int step(int action) {
        if (finished) {
            return 0;
        }

        direction =
            directionForAction(action);

        Point newHead =
            nextPositionForAction(STRAIGHT);

        bool eating =
            newHead == food;

        ++steps;

        if (!inside(newHead) ||
            occupied(newHead, eating)) {

            finished = true;

            return -10;
        }

        snake.push_front(newHead);

        if (eating) {
            score++;
            staleSteps = 0;

            if (static_cast<int>(snake.size()) ==
                BOARD_W * BOARD_H) {

                finished = true;
                return 50;
            }

            spawnFood();

            return 12;
        }

        snake.pop_back();

        staleSteps++;

        // Prevent circling
        if (staleSteps > 500) {
            finished = true;
            return -5;
        }

        // encourages  survival
        return -0.02f;
    }

private:
    void spawnFood() {
        vector<Point> freeCells;

        for (int y = 0; y < BOARD_H; ++y) {
            for (int x = 0; x < BOARD_W; ++x) {

                Point p{x, y};

                if (!occupied(p)) {
                    freeCells.push_back(p);
                }
            }
        }

        if (freeCells.empty()) {
            finished = true;
            return;
        }

        food =
            freeCells[
                RNG.integer(
                    0,
                    static_cast<int>(freeCells.size()) - 1
                )
            ];
    }
};

    
// PLANNER
    

class SafePlanner {
public:

    // Flood fill from a point.

    int reachableArea(
        const SnakeGame& game,
        Point start,
        bool allowTail
    ) const {

        if (!game.inside(start)) {
            return 0;
        }

        vector<vector<bool>> blocked(
            BOARD_H,
            vector<bool>(BOARD_W, false)
        );

        const auto& body =
            game.getSnake();

        for (size_t i = 0; i < body.size(); ++i) {

            // on a non-food move
            if (allowTail &&
                i == body.size() - 1) {
                continue;
            }

            blocked[body[i].y][body[i].x] = true;
        }

        blocked[start.y][start.x] = false;

        vector<vector<bool>> visited(
            BOARD_H,
            vector<bool>(BOARD_W, false)
        );

        deque<Point> q;

        q.push_back(start);
        visited[start.y][start.x] = true;

        static const int dx[4] =
            {1, -1, 0, 0};

        static const int dy[4] =
            {0, 0, 1, -1};

        int count = 0;

        while (!q.empty()) {

            Point p = q.front();
            q.pop_front();

            count++;

            for (int d = 0; d < 4; ++d) {

                Point n{
                    p.x + dx[d],
                    p.y + dy[d]
                };

                if (!game.inside(n)) {
                    continue;
                }

                if (visited[n.y][n.x]) {
                    continue;
                }

                if (blocked[n.y][n.x]) {
                    continue;
                }

                visited[n.y][n.x] = true;
                q.push_back(n);
            }
        }

        return count;
    }

    // head reach tail?

    bool canReachTail(
        const SnakeGame& game,
        Point start
    ) const {

        const auto& body =
            game.getSnake();

        if (body.empty()) {
            return false;
        }

        Point tail = body.back();

        if (start == tail) {
            return true;
        }

        vector<vector<bool>> blocked(
            BOARD_H,
            vector<bool>(BOARD_W, false)
        );
        // tail as target
        for (size_t i = 0; i + 1 < body.size(); ++i) {
            blocked[body[i].y][body[i].x] = true;
        }

        blocked[start.y][start.x] = false;

        vector<vector<bool>> visited(
            BOARD_H,
            vector<bool>(BOARD_W, false)
        );

        deque<Point> q;

        q.push_back(start);
        visited[start.y][start.x] = true;

        static const int dx[4] =
            {1, -1, 0, 0};

        static const int dy[4] =
            {0, 0, 1, -1};

        while (!q.empty()) {

            Point p = q.front();
            q.pop_front();

            if (p == tail) {
                return true;
            }

            for (int d = 0; d < 4; ++d) {

                Point n{
                    p.x + dx[d],
                    p.y + dy[d]
                };

                if (!game.inside(n)) {
                    continue;
                }

                if (visited[n.y][n.x]) {
                    continue;
                }

                if (blocked[n.y][n.x]) {
                    continue;
                }

                visited[n.y][n.x] = true;
                q.push_back(n);
            }
        }

        return false;
    }

    // BFS path to food.

    vector<Point> findFoodPath(
        const SnakeGame& game
    ) const {

        Point start =
            game.getSnake().front();

        Point target =
            game.getFood();

        vector<vector<bool>> blocked(
            BOARD_H,
            vector<bool>(BOARD_W, false)
        );

        const auto& body =
            game.getSnake();

        // do not permanently block 
        for (size_t i = 0; i + 1 < body.size(); ++i) {
            blocked[body[i].y][body[i].x] = true;
        }

        blocked[start.y][start.x] = false;
        blocked[target.y][target.x] = false;

        vector<vector<bool>> visited(
            BOARD_H,
            vector<bool>(BOARD_W, false)
        );

        vector<vector<Point>> parent(
            BOARD_H,
            vector<Point>(
                BOARD_W,
                {-1, -1}
            )
        );

        deque<Point> q;

        q.push_back(start);
        visited[start.y][start.x] = true;

        static const int dx[4] =
            {1, -1, 0, 0};

        static const int dy[4] =
            {0, 0, 1, -1};

        while (!q.empty()) {

            Point p = q.front();
            q.pop_front();

            if (p == target) {
                break;
            }

            for (int d = 0; d < 4; ++d) {

                Point n{
                    p.x + dx[d],
                    p.y + dy[d]
                };

                if (!game.inside(n)) {
                    continue;
                }

                if (visited[n.y][n.x]) {
                    continue;
                }

                if (blocked[n.y][n.x]) {
                    continue;
                }

                visited[n.y][n.x] = true;
                parent[n.y][n.x] = p;

                q.push_back(n);
            }
        }

        if (!visited[target.y][target.x]) {
            return {};
        }

        vector<Point> path;

        Point current = target;

        while (!(current == start)) {
            path.push_back(current);

            Point p =
                parent[current.y][current.x];

            if (p.x < 0) {
                return {};
            }

            current = p;
        }

        path.push_back(start);

        reverse(path.begin(), path.end());

        return path;
    }

    // Analyze action safety

    struct ActionInfo {
        bool legal = false;
        int reachable = 0;
        bool tailReachable = false;
        bool good = false;
    };

    ActionInfo analyzeAction(
        const SnakeGame& game,
        int action
    ) const {

        ActionInfo info;

        if (!game.isImmediateSafe(action)) {
            return info;
        }

        info.legal = true;

        Point next =
            game.nextPositionForAction(action);

        bool eating =
            next == game.getFood();

        info.reachable =
            reachableArea(
                game,
                next,
                !eating
            );

        info.tailReachable =
            canReachTail(game, next);

        int boardArea =
            BOARD_W * BOARD_H;

        float areaRatio =
            static_cast<float>(info.reachable) /
            static_cast<float>(boardArea);

        info.good =
            info.tailReachable &&
            areaRatio >= 0.15f;

        return info;
    }
};

    
// STATE ENCODING
    

static array<float, INPUTS> buildState(
    const SnakeGame& game,
    const SafePlanner& planner
) {
    array<float, INPUTS> state{};

    Point head =
        game.getSnake().front();

    Point food =
        game.getFood();

    // 0-2 : immediate danger for
    //       straight / left / right

    for (int a = 0; a < 3; ++a) {
        state[a] =
            game.isImmediateSafe(a)
                ? 0.0f
                : 1.0f;
    }

    // 3-6 : direction

    state[3 + game.getDirection()] = 1.0f;

    // 7-10 : food relative direction

    if (food.y < head.y)
        state[7] = 1.0f;

    if (food.y > head.y)
        state[8] = 1.0f;

    if (food.x < head.x)
        state[9] = 1.0f;

    if (food.x > head.x)
        state[10] = 1.0f;

    // 11-14 : normalized wall distances

    state[11] =
        static_cast<float>(head.y) /
        BOARD_H;

    state[12] =
        static_cast<float>(BOARD_H - 1 - head.y) /
        BOARD_H;

    state[13] =
        static_cast<float>(head.x) /
        BOARD_W;

    state[14] =
        static_cast<float>(BOARD_W - 1 - head.x) /
        BOARD_W;

    // 15-18 : nearest body distance

    int nearestUp = BOARD_H;
    int nearestDown = BOARD_H;
    int nearestLeft = BOARD_W;
    int nearestRight = BOARD_W;

    const auto& body =
        game.getSnake();

    for (size_t i = 1; i < body.size(); ++i) {

        Point p = body[i];

        if (p.x == head.x) {

            if (p.y < head.y) {
                nearestUp =
                    min(
                        nearestUp,
                        head.y - p.y
                    );
            }

            if (p.y > head.y) {
                nearestDown =
                    min(
                        nearestDown,
                        p.y - head.y
                    );
            }
        }

        if (p.y == head.y) {

            if (p.x < head.x) {
                nearestLeft =
                    min(
                        nearestLeft,
                        head.x - p.x
                    );
            }

            if (p.x > head.x) {
                nearestRight =
                    min(
                        nearestRight,
                        p.x - head.x
                    );
            }
        }
    }

    state[15] =
        min(
            nearestUp,
            BOARD_H
        ) /
        static_cast<float>(BOARD_H);

    state[16] =
        min(
            nearestDown,
            BOARD_H
        ) /
        static_cast<float>(BOARD_H);

    state[17] =
        min(
            nearestLeft,
            BOARD_W
        ) /
        static_cast<float>(BOARD_W);

    state[18] =
        min(
            nearestRight,
            BOARD_W
        ) /
        static_cast<float>(BOARD_W);

    // 19 : number of immediately safe actions

    int safeCount = 0;

    for (int a = 0; a < 3; ++a) {
        if (game.isImmediateSafe(a)) {
            safeCount++;
        }
    }

    state[19] =
        static_cast<float>(safeCount) / 3.0f;

    // 20-22 : reachable area for each action

    for (int a = 0; a < 3; ++a) {

        auto info =
            planner.analyzeAction(
                game,
                a
            );

        state[20 + a] =
            static_cast<float>(info.reachable) /
            static_cast<float>(BOARD_W * BOARD_H);
    }

    // 23 : snake length

    state[23] =
        static_cast<float>(game.getSnake().size()) /
        static_cast<float>(BOARD_W * BOARD_H);

    // 24 : normalized Manhattan distance to food

    int distance =
        abs(food.x - head.x) +
        abs(food.y - head.y);

    state[24] =
        static_cast<float>(distance) /
        static_cast<float>(BOARD_W + BOARD_H);

    return state;
}

    
// ACTION SELECTION
    

struct Decision {
    int action = STRAIGHT;

    string reason;

    vector<SafePlanner::ActionInfo> info;

    vector<Point> foodPath;
};

static Decision chooseAction(
    SnakeGame& game,
    const DQN& network,
    const SafePlanner& planner,
    float epsilon,
    bool training
) {
    Decision decision;

    decision.info.resize(ACTIONS);

    for (int a = 0; a < ACTIONS; ++a) {
        decision.info[a] =
            planner.analyzeAction(game, a);
    }

    decision.foodPath =
        planner.findFoodPath(game);

    // Find legal actions

    vector<int> legal;

    for (int a = 0; a < ACTIONS; ++a) {
        if (decision.info[a].legal) {
            legal.push_back(a);
        }
    }

    // happen when the snake is trapped
    if (legal.empty()) {
        decision.action = STRAIGHT;
        decision.reason = "TRAPPED";
        return decision;
    }

    // Find actions that preserve tail access and space

    vector<int> safe;

    for (int a : legal) {
        if (decision.info[a].good) {
            safe.push_back(a);
        }
    }

    const auto state =
        buildState(
            game,
            planner
        );

    vector<float> q =
        network.predict(state);

    // Exploration.


    if (training &&
        RNG.chance(epsilon)) {

        const vector<int>& choices =
            safe.empty() ? legal : safe;

        decision.action =
            choices[
                RNG.integer(
                    0,
                    static_cast<int>(choices.size()) - 1
                )
            ];

        decision.reason =
            safe.empty()
                ? "RL exploration / legal"
                : "RL exploration / safe";

        return decision;
    }

    // safe food move

    if (decision.foodPath.size() >= 2) {

        Point desired =
            decision.foodPath[1];

        for (int a : legal) {

            Point next =
                game.nextPositionForAction(a);

            if (next == desired) {

                if (decision.info[a].good ||
                    decision.info[a].reachable >
                        BOARD_W * BOARD_H * 0.20f) {

                    decision.action = a;
                    decision.reason =
                        "BFS food path + safety";

                    return decision;
                }
            }
        }
    }

    // combine learned Q value with spatial safety

    float bestScore =
        -numeric_limits<float>::infinity();

    int bestAction =
        legal.front();

    for (int a : legal) {

        float score =
            q[a];

        float area =
            static_cast<float>(
                decision.info[a].reachable
            ) /
            static_cast<float>(
                BOARD_W * BOARD_H
            );

        // Strong preference for preserving space.
        score += 2.5f * area;

        if (decision.info[a].tailReachable) {
            score += 1.5f;
        }

        if (decision.info[a].good) {
            score += 1.0f;
        }

        if (score > bestScore) {
            bestScore = score;
            bestAction = a;
        }
    }

    decision.action = bestAction;

    if (!safe.empty()) {
        decision.reason =
            "RL + safe-space";
    } else {
        decision.reason =
            "RL + legal fallback";
    }

    return decision;
}

    
// EPSILON
    

static float getEpsilon(
    long long trainingSteps
) {
    float t =
        min(
            1.0f,
            static_cast<float>(trainingSteps) /
            static_cast<float>(EPSILON_DECAY_STEPS)
        );

    return
        EPSILON_START +
        t * (EPSILON_END - EPSILON_START);
}

    
// TRAINING STATISTICS
    

struct Statistics {
    deque<int> scores;
    deque<int> steps;
    deque<bool> survived;

    void add(
        int score,
        int stepCount,
        bool survival
    ) {
        scores.push_back(score);
        steps.push_back(stepCount);
        survived.push_back(survival);

        while (scores.size() > 500) {
            scores.pop_front();
            steps.pop_front();
            survived.pop_front();
        }
    }

    double averageScore() const {
        if (scores.empty()) return 0.0;

        double total = 0.0;

        for (int v : scores) {
            total += v;
        }

        return total / scores.size();
    }

    double averageSteps() const {
        if (steps.empty()) return 0.0;

        double total = 0.0;

        for (int v : steps) {
            total += v;
        }

        return total / steps.size();
    }

    double survivalRate() const {
        if (survived.empty()) return 0.0;

        int count = 0;

        for (bool v : survived) {
            if (v) count++;
        }

        return
            100.0 *
            static_cast<double>(count) /
            static_cast<double>(survived.size());
    }
};

    
// EVALUATION
    

struct EvaluationResult {
    double averageScore = 0.0;
    double averageSteps = 0.0;
    double survivalRate = 0.0;
};

static EvaluationResult evaluate(
    DQN& network,
    int games
) {
    SafePlanner planner;

    double scoreTotal = 0.0;
    double stepsTotal = 0.0;
    int survived = 0;

    for (int g = 0; g < games; ++g) {

        SnakeGame game;

        for (int step = 0;
             step < MAX_STEPS_PER_EPISODE &&
             !game.isFinished();
             ++step) {

            Decision d =
                chooseAction(
                    game,
                    network,
                    planner,
                    0.0f,
                    false
                );

            game.step(d.action);
        }

        scoreTotal += game.getScore();
        stepsTotal += game.getSteps();

        if (game.getSteps() >=
            MAX_STEPS_PER_EPISODE) {
            survived++;
        }
    }

    EvaluationResult result;

    result.averageScore =
        scoreTotal / games;

    result.averageSteps =
        stepsTotal / games;

    result.survivalRate =
        100.0 *
        static_cast<double>(survived) /
        static_cast<double>(games);

    return result;
}

    
// VISUALIZATION HELPERS
    

#ifdef _WIN32

static void enableAnsi() {
    HANDLE hOut =
        GetStdHandle(STD_OUTPUT_HANDLE);

    if (hOut == INVALID_HANDLE_VALUE) {
        return;
    }

    DWORD mode = 0;

    if (!GetConsoleMode(hOut, &mode)) {
        return;
    }

    mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;

    SetConsoleMode(hOut, mode);
}

static void clearScreen() {
    cout << "\x1b[2J\x1b[H";
}

static bool keyPressed() {
    return _kbhit() != 0;
}

static char readKey() {
    return static_cast<char>(_getch());
}

#else

static void enableAnsi() {}

static void clearScreen() {
    cout << "\033[2J\033[H";
}

static bool keyPressed() {
    return false;
}

static char readKey() {
    return 0;
}

#endif

    
// DRAW BOARD
    

static void drawGame(
    const SnakeGame& game,
    const DQN& network,
    const SafePlanner& planner,
    const Decision& decision
) {
    clearScreen();

    cout
        << "============================================================\n"
        << "                           SNAKE \n"
        << "============================================================\n";

    cout
        << "Score: "
        << game.getScore()
        << "    Length: "
        << game.getSnake().size()
        << "    Steps: "
        << game.getSteps()
        << "\n";

    cout
        << "Controller: "
        << decision.reason
        << "\n";

    cout
        << "Food path: ";

    if (decision.foodPath.empty()) {
        cout << "none";
    } else {
        cout
            << decision.foodPath.size() - 1
            << " moves";
    }

    cout << "\n\n";

    // Top border.
    cout << "+";
    for (int x = 0; x < BOARD_W; ++x) {
        cout << "--";
    }
    cout << "+\n";

    const auto& body =
        game.getSnake();

    Point food =
        game.getFood();

    for (int y = 0; y < BOARD_H; ++y) {

        cout << "|";

        for (int x = 0; x < BOARD_W; ++x) {

            Point p{x, y};

            string cell = "  ";

            if (p == food) {
                cell = "[]";
            }

            for (size_t i = 0;
                 i < body.size();
                 ++i) {

                if (body[i] == p) {

                    if (i == 0) {
                        cell = "@@";
                    } else {
                        cell = "##";
                    }

                    break;
                }
            }

            cout << cell;
        }

        cout << "|\n";
    }

    cout << "+";

    for (int x = 0; x < BOARD_W; ++x) {
        cout << "--";
    }

    cout << "+\n\n";

    array<float, INPUTS> state =
        buildState(
            game,
            planner
        );

    vector<float> q =
        network.predict(state);

    cout << fixed << setprecision(3);

    cout
        << "Q-values:\n"
        << "  Straight : "
        << setw(8)
        << q[STRAIGHT]
        << (decision.info[STRAIGHT].legal
                ? "  SAFE"
                : "  DEAD")
        << "\n"

        << "  Left     : "
        << setw(8)
        << q[TURN_LEFT]
        << (decision.info[TURN_LEFT].legal
                ? "  SAFE"
                : "  DEAD")
        << "\n"

        << "  Right    : "
        << setw(8)
        << q[TURN_RIGHT]
        << (decision.info[TURN_RIGHT].legal
                ? "  SAFE"
                : "  DEAD")
        << "\n\n";
        cout << "Press Q to stop visualization.\n";
}

    
// VISUAL PLAY
    

static void visualize(
    DQN& network
) {
    SafePlanner planner;

    SnakeGame game;

    cout << "\n";
    cout << "============================================================\n";
    cout << "                 TRAINING COMPLETE\n";
    cout << "============================================================\n\n";

    cout
        << "Starting visual demonstration...\n"
        << "The trained network and safe planner will now play.\n\n";

    this_thread::sleep_for(
        chrono::milliseconds(1200)
    );

    while (true) {

        game.reset();

        while (!game.isFinished()) {

            Decision decision =
                chooseAction(
                    game,
                    network,
                    planner,
                    0.0f,
                    false
                );

            drawGame(
                game,
                network,
                planner,
                decision
            );

            game.step(decision.action);

            for (int i = 0;
                 i < VISUAL_DELAY_MS;
                 i += 10) {

                this_thread::sleep_for(
                    chrono::milliseconds(10)
                );

#ifdef _WIN32
                if (keyPressed()) {
                    char c = readKey();

                    if (c == 'q' ||
                        c == 'Q') {
                        return;
                    }
                }
#endif
            }
        }

        drawGame(
            game,
            network,
            planner,
            Decision{}
        );

        cout
            << "\nGAME OVER\n"
            << "Score: "
            << game.getScore()
            << "\n\n";

        cout
            << "Starting another demonstration in 2 seconds.\n"
            << "Press Q to quit.\n";

        for (int i = 0; i < 200; i += 10) {

            this_thread::sleep_for(
                chrono::milliseconds(10)
            );

#ifdef _WIN32
            if (keyPressed()) {
                char c = readKey();

                if (c == 'q' ||
                    c == 'Q') {
                    return;
                }
            }
#endif
        }
    }
}

    
// TRAINING
    

static void train(
    DQN& online,
    DQN& target
) {
    ReplayBuffer replay;

    SafePlanner planner;

    Statistics stats;

    long long trainingSteps = 0;

    int bestAverageScore = -1;

    int stableEvaluations = 0;

    cout
        << "\n============================================================\n"
        << "                    TRAINING STARTED\n"
        << "============================================================\n\n";

    cout
        << "Episodes: "
        << MAX_EPISODES
        << "\n";

    cout
        << "Maximum steps/episode: "
        << MAX_STEPS_PER_EPISODE
        << "\n";

    cout
        << "Progress is printed every "
        << PRINT_EVERY
        << " episodes.\n\n";

    cout.flush();

    auto startTime =
        chrono::steady_clock::now();

    for (int episode = 1;
         episode <= MAX_EPISODES;
         ++episode) {

        SnakeGame game;

        int episodeSteps = 0;

        float epsilon =
            getEpsilon(trainingSteps);

        while (!game.isFinished() &&
               episodeSteps < MAX_STEPS_PER_EPISODE) {

            array<float, INPUTS> state =
                buildState(
                    game,
                    planner
                );

            Decision decision =
                chooseAction(
                    game,
                    online,
                    planner,
                    epsilon,
                    true
                );

            int action =
                decision.action;

            int reward =
                game.step(action);

            bool done =
                game.isFinished();

            array<float, INPUTS> nextState =
                buildState(
                    game,
                    planner
                );

            Transition transition;

            transition.state = state;
            transition.action = action;
            transition.reward =
                static_cast<float>(reward);
            transition.nextState = nextState;
            transition.done = done;

            replay.push(transition);

            episodeSteps++;
            trainingSteps++;

            // Start learning once there is enough diversity
            if (replay.size() >= BATCH_SIZE) {

                vector<Transition> batch =
                    replay.sample(BATCH_SIZE);

                online.trainBatch(
                    batch,
                    target
                );
            }

            if (trainingSteps %
                TARGET_UPDATE_EVERY == 0) {

                target.copyFrom(online);
            }
        }

        bool survived =
            !game.isFinished() &&
            episodeSteps >= MAX_STEPS_PER_EPISODE;

        stats.add(
            game.getScore(),
            episodeSteps,
            survived
        );

        // LIVE OUTPUT

        if (episode % PRINT_EVERY == 0) {

            auto now =
                chrono::steady_clock::now();

            double seconds =
                chrono::duration<double>(
                    now - startTime
                ).count();

            double epsPerSecond =
                seconds > 0.0
                    ? episode / seconds
                    : 0.0;

            cout
                << "Episode "
                << setw(6)
                << episode
                << " | Score "
                << setw(4)
                << game.getScore()
                << " | Steps "
                << setw(5)
                << episodeSteps
                << " | Avg "
                << fixed
                << setprecision(2)
                << setw(6)
                << stats.averageScore()
                << " | Survival "
                << setw(6)
                << setprecision(1)
                << stats.survivalRate()
                << "%"
                << " | Eps "
                << setprecision(3)
                << epsilon
                << " | "
                << setprecision(1)
                << epsPerSecond
                << " ep/s"
                << "\n";

            cout.flush();
        }

        // PERIODIC EVALUATION

        if (episode % EVALUATE_EVERY == 0) {

            EvaluationResult result =
                evaluate(
                    online,
                    EVALUATION_GAMES
                );

            cout
                << "\n"
                << "------------------------------------------------------------\n"
                << "EVALUATION @ EPISODE "
                << episode
                << "\n"
                << "Average food: "
                << fixed
                << setprecision(2)
                << result.averageScore
                << "\n"
                << "Average steps: "
                << result.averageSteps
                << "\n"
                << "Survival to step limit: "
                << result.survivalRate
                << "%\n"
                << "------------------------------------------------------------\n\n";

            cout.flush();

            // Save whenever average score improves.
            if (result.averageScore >
                bestAverageScore) {

                bestAverageScore =
                    static_cast<int>(
                        result.averageScore
                    );

                online.save(
                    "snake_best_model.txt"
                );

                cout
                    << "Saved improved model -> "
                    << "snake_best_model.txt\n\n";
            }


            if (result.survivalRate >= 80.0 &&
                result.averageScore >= 8.0) {

                stableEvaluations++;

                cout
                    << "Stability check: "
                    << stableEvaluations
                    << "/5 passed.\n\n";

            } else {
                stableEvaluations = 0;
            }

            // Five consecutive evaluations
            if (stableEvaluations >= 5) {

                cout
                    << "\n============================================================\n"
                    << "             TRAINING STABILIZED EARLY\n"
                    << "============================================================\n\n";

                cout
                    << "The agent maintained the required evaluation\n"
                    << "threshold for five consecutive evaluations.\n\n";

                break;
            }
        }
    }

    // Final synchronization.
    target.copyFrom(online);

    online.save(
        "snake_final_model.txt"
    );

    cout
        << "\n============================================================\n"
        << "                    TRAINING FINISHED\n"
        << "============================================================\n\n";

    cout
        << "Final model saved as:\n"
        << "  snake_final_model.txt\n\n";

    cout
        << "Best model saved as:\n"
        << "  snake_best_model.txt\n\n";
}

    
// MAIN


int main() {

#ifdef _WIN32
    enableAnsi();
#endif

    cout.setf(ios::unitbuf);

    cout
        << "============================================================\n"
        << "                 SNAKE TRAINING SYSTEM\n"
        << "============================================================\n\n";

    cout
        << "Mode:              RL + SAFE PATH PLANNER\n" << "Board:             "<< BOARD_W
        << " x "<< BOARD_H << "\n"<< "State:             " << " features\n" << "Network:           "
        << INPUTS<< " -> "<< H1 << " -> " << H2 << " -> " << H3<< " -> " << ACTIONS << "\n"
        << "Replay memory:     " << REPLAY_CAPACITY<< "\n"<< "Algorithm:         Double DQN\n"
        << "Planner:            BFS + flood fill + tail safety\n"<< "Exploration:       SAFE epsilon-greedy\n"
        << "Training limit:    "<< MAX_EPISODES<< " episodes\n" << "Episode limit:     "<< MAX_STEPS_PER_EPISODE
        << " steps\n"
        << "Visualization:     AFTER TRAINING\n\n";

    cout
        << "The training loop will print progress every "
        << PRINT_EVERY
        << " episodes.\n";


    DQN online;
    DQN target;

    target.copyFrom(online);

    train(
        online,
        target
    );
   // VISUAL DEMONSTRATION

    visualize(online);

    cout
        << "\nVisualization ended.\n";


    return 0;
}