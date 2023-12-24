from audioop import mul
from concurrent.futures import process
from PyQt6 import QtGui, QtCore, QtWidgets, uic
import numpy as np
import sys, os, time, math

path = os.path.dirname(os.path.realpath(__file__))
#Engine
class Engine():
    def __init__(self):
        self.time_out = None
        self.negamax_best_move = None
        #initialize transposition table
        self.transposition_table = {}
        #initialize zobrist hash table
        self.zobrist_white = np.random.randint(2**63, size=(8, 8), dtype=np.int64)
        self.zobrist_black = np.random.randint(2**63, size=(8, 8), dtype=np.int64)
    def get_best_move(self, board, depth, turn):
        self.board = board
        self.depth = depth
        self.turn = turn
        self.eval = None
        self.time_out = time.time() + 5
        self.cur_root = 1
        while time.time() <= self.time_out and self.cur_root <= self.depth:
            self.negamax(self.board, self.cur_root, self.turn)
            self.cur_root += 1
        return self.negamax_best_move
    def negamax(self, board, depth, turn, alpha = -math.inf, beta = math.inf):
        if time.time() > self.time_out:
            return self.eval
        lookup = self.transposition_table_lookup(board)
        if lookup is not None:
            if lookup["depth"] >= self.depth and lookup["turn"] == self.turn:
                self.negamax_best_move = lookup["move"]
                self.eval = lookup["eval"]
                return lookup["eval"]
            if lookup["depth"] >= depth:
                if lookup["turn"] == turn:
                    return lookup["eval"]
                else:
                    return -lookup["eval"]
        legal_moves = self.search_legal_moves(board, turn)
        if depth == 0 or len(legal_moves) == 0:
            if turn == 1:
                return self.eval_position(board)
            else:
                return -self.eval_position(board)
        max_eval = -math.inf
        for move in legal_moves:
            posible_board_position = self.posible_position(board, move, turn)
            eval = -self.negamax(posible_board_position, depth - 1, 3-turn, -beta, -alpha)
            if eval > max_eval:
                max_eval = eval
                if depth == self.cur_root:
                    self.negamax_best_move = move
                    self.eval = max_eval
            alpha = max(alpha, eval)
            if alpha >= beta:
                break
        self.transposition_table_store(board, depth, turn, max_eval, move)
        return max_eval
    def get_eval(self, fromBotMove = False):
        if fromBotMove:
            return self.eval
        else:
            return self.negamax(self.board, self.depth, self.turn)
    def get_depth(self):
        return self.cur_root
    def search_legal_moves(self, board, color):
        #move = (i, j)
        #find legal moves
        legal_moves = []
        if color == 1:
            other_color = 2
        else:
            other_color = 1
        color_pos = np.where(board == color)
        for i, j in zip(color_pos[0], color_pos[1]):
            #search legal moves in each direction
            for direction in [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),          (1, 0),
                (-1, 1), (0, 1), (1, 1)
            ]:
                x, y = i, j
                while True:
                    x += direction[0]
                    y += direction[1]
                    if x < 0 or x > 7 or y < 0 or y > 7:
                        break
                    if board[x, y] == 0:
                        break
                    if board[x, y] == other_color: #if there is a piece of the other color, move one more square in the same direction. Also check if that square is within the board and empty.
                        if (x + direction[0] >= 0 and x + direction[0] <= 7 and y + direction[1] >= 0 and y + direction[1] <= 7):
                            if board[x + direction[0], y + direction[1]] == 0:
                                x += direction[0]
                                y += direction[1]
                                legal_moves.append((x, y))
                                break
        legal_moves = list(dict.fromkeys(legal_moves))
        return legal_moves
    def flip_pieces(self, board, move, color):
        #mode 0: normal, 1: eval in logic_board
        #flip pieces between two 'color' pieces
        if color == 1:
            other_color = 2
        else:
            other_color = 1
        #flip pieces in each direction
        for direction in [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),          (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ]:
            x, y = move[0], move[1]
            while True:
                x += direction[0]
                y += direction[1]
                if x < 0 or x > 7 or y < 0 or y > 7:
                    break
                if board[x, y] == 0  or board[x, y] == color:
                    break
                if board[x, y] == other_color:
                    if (x + direction[0] >= 0 and x + direction[0] <= 7 and y + direction[1] >= 0 and y + direction[1] <= 7):
                        if board[x + direction[0], y + direction[1]] == other_color:
                            continue
                        if board[x + direction[0], y + direction[1]] == 0:
                            break
                        else:
                            x += direction[0]
                            y += direction[1]
                            while True:
                                x -= direction[0]
                                y -= direction[1]
                                if board[x, y] == color:
                                    break
                                else:
                                    board[x, y] = color    
                            break
    def posible_position(self, board, move, turn):
        posible_board_position = board.copy()
        posible_board_position[move[0], move[1]] = turn
        self.flip_pieces(posible_board_position, move, turn)
        return posible_board_position
    def eval_position(self, board):
        white_score = 0
        black_score = 0
        for i in range(8):
            for j in range(8):
                if board[i, j] == 1:
                    white_score += 1
                    if i == 0 or i == 7 or j == 0 or j == 7:
                        white_score += 3
                    if i == 3 and j == 3 or i == 4 and j == 4 or i == 3 and j == 4 or i == 4 and j == 3:
                        white_score += 4
                    if i == 0 and j == 0 or i == 0 and j == 7 or i == 7 and j == 0 or i == 7 and j == 7:
                        white_score += 6
                elif board[i, j] == 2:
                    black_score += 1
                    if i == 0 or i == 7 or j == 0 or j == 7:
                        black_score += 3
                    if i == 3 and j == 3 or i == 4 and j == 4 or i == 3 and j == 4 or i == 4 and j == 3:
                        black_score += 4
                    if i == 0 and j == 0 or i == 0 and j == 7 or i == 7 and j == 0 or i == 7 and j == 7:
                        black_score += 6
        return (white_score - black_score)/2
    def transposition_table_store(self, board, depth, turn, eval, move):
        key = self.zobrist_key(board)
        self.transposition_table[key] = {"depth" : depth, 
                                         "turn" : turn, 
                                         "eval" : eval,
                                         "move" : move}
    def transposition_table_lookup(self, board):
        key = self.zobrist_key(board)
        if key in self.transposition_table:
            return self.transposition_table[key]
        else:
            return None
    def zobrist_key(self, board):
        key = 0
        for i in range(8):
            for j in range(8):
                if board[i, j] == 1:
                    key ^= self.zobrist_white[i, j]
                elif board[i, j] == 2:
                    key ^= self.zobrist_black[i, j]
        return key

class Menu(QtWidgets.QMainWindow):
    def __init__(self):
        super(Menu, self).__init__()
        uic.loadUi(path + r'\UI\menu.ui', self)
        self.setWindowTitle('Menu')
        self.setFixedSize(self.size())
#Result Window
class Result_popup(QtWidgets.QMainWindow):
    def __init__(self):
        super(Result_popup, self).__init__()
        uic.loadUi(path + r'\UI\result_popup.ui', self)
        self.setWindowTitle('Result')
        self.setFixedSize(self.size())
#Main Window
class Othello(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        #initialize logic board. 0: empty, 1: white, 2: black
        self.logic_board = np.zeros((8, 8))
        self.turn = 1
        #initialize Engine
        self.engine = Engine()
        #initialize board UI
        uic.loadUi(path + r"\UI\Othello.ui", self)
        self.setWindowTitle('Othello')
        self.newGame_button.clicked.connect(self.newgame_button_on_click)
        self.menu_button.clicked.connect(self.menu_button_onclick)
        self.board_img.mousePressEvent = self.player_move
        self.one_square_width = self.board_img.width() / 7.95
        self.one_square_height = self.board_img.height() / 8
        self.square_coor = self.get_square_coor()
        self.empty_board = self.board_img.pixmap()
        self.draw_starting_pieces()
        #initialize menu UI
        self.menu = Menu()
        self.menu.comboBox_bot_toggle.currentIndexChanged.connect(self.comboBox_bot_toggle_on_change)
        self.menu.comboBox_bot_difficulty.currentIndexChanged.connect(self.comboBox_bot_difficulty_on_change)
        self.is_vs_bot = True
        self.depth_ = 1
        #initialize result UI
        self.result_popup = None
    #Event handler    
    def resizeEvent(self, event):
        self.one_square_width = self.board_img.width() / 7.95
        self.one_square_height = self.board_img.height() / 8
        self.square_coor = self.get_square_coor()
    def newgame_button_on_click(self):
        self.board_img.setPixmap(self.empty_board)
        self.logic_board = np.zeros((8, 8))
        self.draw_starting_pieces()
    def menu_button_onclick(self):
        if (self.menu.isVisible()):
            self.menu.hide()
        else:
            self.menu.show()
        print("menu clicked")
    def comboBox_bot_toggle_on_change(self):
        if self.menu.comboBox_bot_toggle.currentText() == "Off":
            self.is_vs_bot = False
        else:
            self.is_vs_bot = True
    def comboBox_bot_difficulty_on_change(self):
        if self.menu.comboBox_bot_difficulty.currentText() == "Easy":
            self.depth_ = 2
        elif self.menu.comboBox_bot_difficulty.currentText() == "Medium":
            self.depth_ = 5
        elif self.menu.comboBox_bot_difficulty.currentText() == "Hard":
            self.depth_ = 7
        elif self.menu.comboBox_bot_difficulty.currentText() == "Impossible":
            self.depth_ = 15
    #Draw pieces
    def draw_pieces(self, coor_x, coor_y, color):
        if color == 1:
            color = QtGui.QColor(255, 255, 255)
        else:
            color = QtGui.QColor(0, 0, 0)
        #draw a circle at square where mouse clicks
        pixmap = QtGui.QPixmap(self.board_img.pixmap())
        pixmap = pixmap.scaled(self.board_img.size())
        painter = QtGui.QPainter()
        painter.begin(pixmap)
        painter.setBrush(color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setRenderHints(painter.RenderHint.Antialiasing)
        painter.drawEllipse(self.square_coor[coor_x][coor_y][0]+5, self.square_coor[coor_x][coor_y][1]+5, self.one_square_width-10, self.one_square_height-10)
        painter.end()
        self.board_img.setPixmap(pixmap)
        self.board_img.repaint()
    def draw_starting_pieces(self):
        #draw starting pieces
        pixmap = QtGui.QPixmap(self.board_img.pixmap())
        pixmap = pixmap.scaled(self.board_img.size())
        painter = QtGui.QPainter()
        painter.begin(pixmap)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setRenderHints(painter.RenderHint.Antialiasing)
        #White pieces
        self.logic_board[3, 3] = 1
        self.logic_board[4, 4] = 1
        painter.setBrush(QtGui.QColor(255, 255, 255))
        painter.drawEllipse(self.square_coor[3][3][0]+5, self.square_coor[3][3][1]+5, self.one_square_width-10, self.one_square_height-10)
        painter.drawEllipse(self.square_coor[4][4][0]+5, self.square_coor[4][4][1]+5, self.one_square_width-10, self.one_square_height-10)
        #Black pieces
        self.logic_board[3, 4] = 2
        self.logic_board[4, 3] = 2
        painter.setBrush(QtGui.QColor(0, 0, 0))
        painter.drawEllipse(self.square_coor[3][4][0]+5, self.square_coor[3][4][1]+5, self.one_square_width-10, self.one_square_height-10)
        painter.drawEllipse(self.square_coor[4][3][0]+5, self.square_coor[4][3][1]+5, self.one_square_width-10, self.one_square_height-10)
        painter.end()
        self.board_img.setPixmap(pixmap)   
    #Game turn
    def player_move(self, event):
        x = event.position().x()
        y = event.position().y()
        self.mouse_pos = (x, y)
        #check square_coor
        for i in range(8):
            for j in range(8):
                if self.square_coor[i, j, 0] < x < self.square_coor[i, j, 0] + self.one_square_width and self.square_coor[i, j, 1] < y < self.square_coor[i, j, 1] + self.one_square_height:
                    move = (i, j)
                    if self.is_a_legal_move(self.logic_board, move):
                        self.draw_pieces(i, j, self.turn)
                        self.logic_board[i, j] = self.turn
                        self.flip_pieces(self.logic_board, move, self.turn)
                        print("\nPlayer move:", i, j)
                        if self.is_vs_bot:
                            self.turn = 3 - self.turn
                            self.board_img.mousePressEvent = None
                            self.bot_move()
                        else:
                            self.turn = 3 - self.turn
                            self.check_if_game_over()
                        return
    def bot_move(self):
        time.sleep(0.75)
        legal_moves = self.search_legal_moves(self.logic_board, self.turn)
        if len(legal_moves) == 1:
            best_move = legal_moves[0]
        elif len(legal_moves) > 1:
            best_move = self.engine.get_best_move(self.logic_board, self.depth_, self.turn)
        else:
            best_move = None
        if best_move is not None:
            print("Bot move:", best_move)
            self.draw_pieces(best_move[0], best_move[1], self.turn)
            self.logic_board[best_move[0], best_move[1]] = self.turn
            self.flip_pieces(self.logic_board, best_move, self.turn)
            self.turn = 3 - self.turn
        #Move info
        eval = self.engine.get_eval(True)
        depth = self.engine.get_depth()
        print("Eval:", eval)
        print("Depth:", depth)

        self.check_if_game_over()
        self.board_img.mousePressEvent = self.player_move
    #Game logic
    def get_square_coor(self):
        square_coor = np.zeros((8, 8, 2))
        for i in range(8):
            for j in range(8):
                square_coor[i, j, 0] = i * self.one_square_width 
                square_coor[i, j, 1] = j * self.one_square_height
        return square_coor
    def is_a_legal_move(self, board, move):
        is_a_legal_move = False
        legal_moves = self.search_legal_moves(self.logic_board, self.turn)
        if board[move[0], move[1]] == 0 and move in legal_moves:
            is_a_legal_move = True
        return is_a_legal_move
    def search_legal_moves(self, board, color):
        #move = (i, j)
        #find legal moves
        legal_moves = []
        if color == 1:
            other_color = 2
        else:
            other_color = 1
        color_pos = np.where(board == color)
        for i, j in zip(color_pos[0], color_pos[1]):
            #search legal moves in each direction
            for direction in [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),          (1, 0),
                (-1, 1), (0, 1), (1, 1)
            ]:
                x, y = i, j
                while True:
                    x += direction[0]
                    y += direction[1]
                    if x < 0 or x > 7 or y < 0 or y > 7:
                        break
                    if board[x, y] == 0:
                        break
                    if board[x, y] == other_color: #if there is a piece of the other color, move one more square in the same direction. Also check if that square is within the board and empty.
                        if (x + direction[0] >= 0 and x + direction[0] <= 7 and y + direction[1] >= 0 and y + direction[1] <= 7):
                            if board[x + direction[0], y + direction[1]] == 0:
                                x += direction[0]
                                y += direction[1]
                                legal_moves.append((x, y))
                                break
        legal_moves = list(dict.fromkeys(legal_moves))
        return legal_moves
    def flip_pieces(self, board, move, color):
        #mode 0: normal, 1: eval in logic_board
        #flip pieces between two 'color' pieces
        if color == 1:
            other_color = 2
        else:
            other_color = 1
        #flip pieces in each direction
        for direction in [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),          (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ]:
            x, y = move[0], move[1]
            while True:
                x += direction[0]
                y += direction[1]
                if x < 0 or x > 7 or y < 0 or y > 7:
                    break
                if board[x, y] == 0  or board[x, y] == color:
                    break
                if board[x, y] == other_color:
                    if (x + direction[0] >= 0 and x + direction[0] <= 7 and y + direction[1] >= 0 and y + direction[1] <= 7):
                        if board[x + direction[0], y + direction[1]] == other_color:
                            continue
                        if board[x + direction[0], y + direction[1]] == 0:
                            break
                        else:
                            x += direction[0]
                            y += direction[1]
                            while True:
                                x -= direction[0]
                                y -= direction[1]
                                if board[x, y] == color:
                                    break
                                else:
                                    self.draw_pieces(x, y, color)
                                    board[x, y] = color    
                            break
    def check_if_game_over(self):
        legal_moves = self.search_legal_moves(self.logic_board, self.turn)
        if len(legal_moves) == 0:
            white = np.where(self.logic_board == 1)[0].__len__()
            black = np.where(self.logic_board == 2)[0].__len__()
            r = white - black
            if r > 0:
                if self.result_popup == None:
                    self.result_popup = Result_popup()
                    self.result_popup.label.setText("White wins!")
                    self.result_popup.show()
                else:
                    self.result_popup = None
            elif r < 0:
                if self.result_popup == None:
                    self.result_popup = Result_popup()
                    self.result_popup.label.setText("Black wins!")
                    self.result_popup.show()
                else:
                    self.result_popup = None
            else:
                if self.result_popup == None:
                    self.result_popup = Result_popup()
                    self.result_popup.label.setText("Draw!")
                    self.result_popup.show()
                else:
                    self.result_popup = None
            return True
        else:
            return False 
    def print_logic_board(self):
        print(np.flip(np.rot90(self.logic_board, -1), 1))
    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = Othello()
    ex.show()
    sys.exit(app.exec())