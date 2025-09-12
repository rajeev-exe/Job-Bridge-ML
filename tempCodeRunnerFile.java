import java.util.*;

public class SofaProblem {
    static class State {
        int r, c;
        char orientation; // 'H' for horizontal, 'V' for vertical
        int steps;

        State(int r, int c, char orientation, int steps) {
            this.r = r;
            this.c = c;
            this.orientation = orientation;
            this.steps = steps;
        }

        String getKey() {
            return r + "," + c + "," + orientation;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int M = sc.nextInt();
        int N = sc.nextInt();
        char[][] grid = new char[M][N];
        
        // Read the grid
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                grid[i][j] = sc.next().charAt(0);
            }
        }
        
        // Find sofa's initial position and orientation
        int startR = -1, startC = -1;
        char startOrientation = ' ';
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (grid[i][j] == 's') {
                    startR = i;
                    startC = j;
                    // Check if sofa is horizontal or vertical
                    if (j + 1 < N && grid[i][j + 1] == 's') {
                        startOrientation = 'H';
                    } else if (i + 1 < M && grid[i + 1][j] == 's') {
                        startOrientation = 'V';
                    }
                    break;
                }
            }
            if (startR != -1) break;
        }

        // Find destination position
        int destR = -1, destC = -1;
        char destOrientation = ' ';
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (grid[i][j] == 'S') {
                    destR = i;
                    destC = j;
                    // Check if destination is horizontal or vertical
                    if (j + 1 < N && grid[i][j + 1] == 'S') {
                        destOrientation = 'H';
                    } else if (i + 1 < M && grid[i + 1][j] == 'S') {
                        destOrientation = 'V';
                    }
                    break;
                }
            }
            if (destR != -1) break;
        }

        // Run BFS to find minimum steps
        int result = bfs(grid, M, N, startR, startC, startOrientation, destR, destC, destOrientation);
        System.out.println(result == -1 ? "Impossible" : result);
        sc.close();
    }

    static int bfs(char[][] grid, int M, int N, int startR, int startC, char startOrientation,
                  int destR, int destC, char destOrientation) {
        Queue<State> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        
        State start = new State(startR, startC, startOrientation, 0);
        queue.offer(start);
        visited.add(start.getKey());

        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // Up, Down, Left, Right

        while (!queue.isEmpty()) {
            State curr = queue.poll();
            int r = curr.r, c = curr.c;
            char orient = curr.orientation;
            int steps = curr.steps;

            // Check if current state is the destination
            if (r == destR && c == destC && orient == destOrientation) {
                return steps;
            }

            // Try moving in all four directions
            for (int[] dir : directions) {
                int newR = r + dir[0];
                int newC = c + dir[1];
                if (isValidMove(grid, M, N, newR, newC, orient)) {
                    State next = new State(newR, newC, orient, steps + 1);
                    if (!visited.contains(next.getKey())) {
                        visited.add(next.getKey());
                        queue.offer(next);
                    }
                }
            }

            // Try rotating (clockwise and counterclockwise)
            if (canRotate(grid, M, N, r, c, orient)) {
                char newOrient = (orient == 'H') ? 'V' : 'H';
                State next = new State(r, c, newOrient, steps + 1);
                if (!visited.contains(next.getKey())) {
                    visited.add(next.getKey());
                    queue.offer(next);
                }
            }
        }
        
        return -1; // No path found
    }

    static boolean isValidMove(char[][] grid, int M, int N, int r, int c, char orient) {
        // Check boundaries
        if (orient == 'H') {
            if (r < 0 || r >= M || c < 0 || c + 1 >= N) return false;
            // Check if cells are free ('0' or 'S')
            return (grid[r][c] == '0' || grid[r][c] == 'S') &&
                   (grid[r][c + 1] == '0' || grid[r][c + 1] == 'S');
        } else { // Vertical
            if (r < 0 || r + 1 >= M || c < 0 || c >= N) return false;
            return (grid[r][c] == '0' || grid[r][c] == 'S') &&
                   (grid[r + 1][c] == '0' || grid[r + 1][c] == 'S');
        }
    }

    static boolean canRotate(char[][] grid, int M, int N, int r, int c, char orient) {
        // Check if 2x2 area around sofa is free
        int r2 = (orient == 'H') ? r : r + 1;
        int c2 = (orient == 'H') ? c + 1 : c;
        // Ensure 2x2 area is within bounds
        if (r < 0 || r + 1 >= M || c < 0 || c + 1 >= N) return false;
        // Check all four cells in the 2x2 area
        for (int i = r; i <= r + 1; i++) {
            for (int j = c; j <= c + 1; j++) {
                if (grid[i][j] == 'H') return false;
            }
        }
        return true;
    }
}